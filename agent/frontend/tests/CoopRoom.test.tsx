import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import CoopRoom from '../src/coop/CoopRoom';
import * as api from '../src/coop/api';
import type { Room } from '../src/coop/api';
import { setLanguage } from '../src/i18n';

// The assertions use the Chinese UI strings.
setLanguage('zh');

vi.mock('../src/coop/api', async (original) => ({ ...await original<typeof api>(),
  getRoom: vi.fn(), watchRoom: vi.fn(), startImmersive: vi.fn(), advanceImmersive: vi.fn(),
  stopImmersive: vi.fn(), submitAction: vi.fn(), rollback: vi.fn(), listRooms: vi.fn() }));
vi.mock('../src/api', () => ({ BASE: '/', mediaUrl: (kind: string, clip: string) => `/${kind}/${clip}`,
  getHealth: async () => ({ tts: false }) }));
vi.mock('../src/tts', () => ({ loadTts: () => false, saveTts: vi.fn(), speak: vi.fn(), stopSpeaking: vi.fn() }));

const room = { code: 'ABCDEF123456', mode: 'local', title: '测试山洞', host_id: 'a', world_id: 'test-world',
  round: 1, revision: 1, status: 'waiting', scene_name: '', scene_ref: '', setting: '山洞探索', visual_style: '电影',
  capacity: 2, rounds: [], opening_node: { id: 'one', clip: 'one.mp4' }, generation_enabled: true,
  auto_generate: true, ai_mode: 'offline', error: '', progress: '',
  members: ['a', 'b'].map((user_id) => ({ user_id, name: user_id, character: `角色${user_id}`,
    description: '', ref: 'ref.jpg', action: '', ready: false, subject_id: user_id })),
  intro: { status: 'ready', narration: '', objective: '离开山洞', world_event: '巨龙抬起头准备观察洞口', error: '',
    suggestions: [{ user_id: 'a', action: '沿左侧岩壁前进' }, { user_id: 'b', action: '沿右侧岩壁前进' }] },
} as Room;

beforeEach(() => {
  HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  HTMLDialogElement.prototype.close = function () { this.open = false; };
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
  vi.mocked(api.getRoom).mockResolvedValue(room);
  vi.mocked(api.watchRoom).mockReturnValue(() => undefined);
  vi.mocked(api.startImmersive).mockResolvedValue(room);
  vi.mocked(api.advanceImmersive).mockResolvedValue({ ...room, revision: 2, status: 'generating' });
  vi.mocked(api.stopImmersive).mockResolvedValue({ ...room, revision: 3 });
  vi.mocked(api.submitAction).mockResolvedValue({ ...room, revision: 4 });
});
afterEach(async () => { await act(async () => cleanup()); vi.restoreAllMocks(); vi.clearAllMocks(); });

it('clears the old user’s lobby immediately and ignores their delayed response', async () => {
  let resolveOld!: (rooms: api.RoomSummary[]) => void;
  let resolveNew!: (rooms: api.RoomSummary[]) => void;
  vi.mocked(api.listRooms).mockResolvedValueOnce([room])
    .mockReturnValueOnce(new Promise(resolve => { resolveOld = resolve; }))
    .mockReturnValueOnce(new Promise(resolve => { resolveNew = resolve; }));
  const props = { code: '', theme: 'mist' as const, onThemeChange: () => undefined,
    onExit: () => undefined, onSwitchUser: () => undefined };
  const user = (id: string) => ({ id, name: id, locked: false, created: '' });
  const ui = render(<CoopRoom {...props} user={user('a')} />);
  await ui.findByRole('button', { name: new RegExp(`^${room.title}`) });
  ui.rerender(<CoopRoom {...props} user={user('b')} />);
  expect(ui.queryByRole('button', { name: new RegExp(`^${room.title}`) })).toBeNull();
  ui.rerender(<CoopRoom {...props} user={user('c')} />);
  await act(async () => resolveNew([{ ...room, code: 'CROOM', title: 'C 的世界' }]));
  await act(async () => resolveOld([{ ...room, title: 'B 的世界' }]));
  expect(ui.getByRole('button', { name: /C 的世界/ })).toBeTruthy();
  expect(ui.queryByRole('button', { name: /B 的世界/ })).toBeNull();
});

it('stops immersive generation before switching the login cookie', async () => {
  let resolveStop!: (room: Room) => void;
  vi.mocked(api.stopImmersive).mockReturnValueOnce(new Promise(resolve => { resolveStop = resolve; }));
  const switchUser = vi.fn();
  const ui = render(<CoopRoom code={room.code} user={{ id: 'a', name: 'Alice', created: '' }}
    theme="mist" onThemeChange={() => undefined} onExit={() => undefined} onSwitchUser={switchUser} />);
  await ui.findByText(room.intro!.world_event!);
  fireEvent.click(ui.getByRole('button', { name: '沉浸模式' }));
  await waitFor(() => expect(api.advanceImmersive).toHaveBeenCalled());
  fireEvent.click(ui.getByRole('button', { name: '切换身份：Alice' }));
  await waitFor(() => expect(api.stopImmersive).toHaveBeenCalled());
  expect(switchUser).not.toHaveBeenCalled();
  await act(async () => resolveStop(room));
  expect(switchUser).toHaveBeenCalledTimes(1);
});

it('confirms rollback of the latest shared clip and restores both action drafts', async () => {
  const current = { ...room, round: 2, rounds: [{ round: 1, inputs: [], merged_prompt: '',
    node: { ...room.opening_node!, id: 'two', clip: 'two.mp4' } }] };
  vi.mocked(api.getRoom).mockResolvedValue(current);
  vi.mocked(api.rollback).mockResolvedValue({ ...room, revision: 2,
    members: room.members.map(p => ({ ...p, action: `重试${p.user_id}` })) });
  const { getByRole, getByLabelText, container } = render(<CoopRoom code={room.code}
    user={{ id: 'a', name: '房主', locked: false, created: '' }} theme="mist" onThemeChange={() => undefined}
    onExit={() => undefined} onSwitchUser={() => undefined} />);
  await waitFor(() => expect(container.querySelector('video')?.getAttribute('src')).toContain('two.mp4'));
  fireEvent.click(getByRole('button', { name: '撤回最新一段' }));
  expect(api.rollback).not.toHaveBeenCalled();
  fireEvent.click(getByRole('button', { name: '确认撤回' }));
  await waitFor(() => expect(api.rollback).toHaveBeenCalledWith(room.code, 'two'));
  await waitFor(() => expect(container.querySelector('video')?.getAttribute('src')).toContain('one.mp4'));
  expect((getByLabelText('玩家 A 本轮动作') as HTMLTextAreaElement).value).toBe('重试a');
  fireEvent.click(getByRole('button', { name: '切换到玩家 B：角色b' }));
  expect((getByLabelText('玩家 B 本轮动作') as HTMLTextAreaElement).value).toBe('重试b');
});

it('hides shared rollback from non-host players', async () => {
  const { queryByRole, getByText } = render(<CoopRoom code={room.code}
    user={{ id: 'b', name: '队友', locked: false, created: '' }} theme="mist" onThemeChange={() => undefined}
    onExit={() => undefined} onSwitchUser={() => undefined} />);
  await waitFor(() => expect(getByText(room.intro!.world_event!)).toBeTruthy());
  expect(queryByRole('button', { name: '撤回最新一段' })).toBeNull();
});

it('shows chapter preparation and preserves the current video', async () => {
  vi.mocked(api.getRoom).mockResolvedValue({ ...room, intro: undefined,
    chapter_transition: { pending: true, status: 'assets' }, generation_enabled: false });
  const { getByText, getByRole, container } = render(<CoopRoom code={room.code}
    user={{ id: 'a', name: '测试用户', locked: false, created: '' }} theme="mist" onThemeChange={() => undefined}
    onExit={() => undefined} onSwitchUser={() => undefined} />);
  await waitFor(() => expect(getByText('正在准备下一关')).toBeTruthy());
  expect((getByRole('button', { name: '发送玩家 A 的动作' }) as HTMLButtonElement).disabled).toBe(true);
  expect(container.querySelector('video')?.getAttribute('src')).toContain('one.mp4');
  expect(api.submitAction).not.toHaveBeenCalled();
});

it('restores the forecast and both usable suggestions after cancelling immersive generation', async () => {
  const { getByRole, getByLabelText, getByText, container } = render(<CoopRoom code={room.code}
    user={{ id: 'a', name: '测试用户', locked: false, created: '' }} theme="mist" onThemeChange={() => undefined}
    onExit={() => undefined} onSwitchUser={() => undefined} />);
  await waitFor(() => expect(getByText(room.intro!.world_event!)).toBeTruthy());
  fireEvent.click(getByRole('button', { name: '沉浸模式' }));
  await waitFor(() => expect(api.advanceImmersive).toHaveBeenCalledTimes(1));
  expect(container.querySelector('#app')?.classList.contains('immersive')).toBe(true);
  // This exit control must live outside the workbar hidden by immersive CSS.
  const exit = getByRole('button', { name: '退出沉浸', exact: true });
  expect(exit.closest('.coop-workbar')).toBeNull();
  fireEvent.click(exit);
  await waitFor(() => expect(api.stopImmersive).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(container.querySelector('.coop-generating')).toBeNull());
  expect(container.querySelector('#app')?.classList.contains('immersive')).toBe(false);
  expect(getByText(room.intro!.world_event!)).toBeTruthy();
  const adoptA = getByRole('button', { name: '采用建议：沿左侧岩壁前进' });
  expect((adoptA as HTMLButtonElement).disabled).toBe(false);
  fireEvent.click(adoptA);
  expect((getByLabelText('玩家 A 本轮动作') as HTMLTextAreaElement).value).toBe('沿左侧岩壁前进');
  fireEvent.click(getByRole('button', { name: '发送玩家 A 的动作' }));
  await waitFor(() => expect(api.submitAction).toHaveBeenCalledTimes(1));
  fireEvent.click(getByRole('button', { name: '切换到玩家 B：角色b' }));
  fireEvent.click(getByRole('button', { name: '采用建议：沿右侧岩壁前进' }));
  expect((getByLabelText('玩家 B 本轮动作') as HTMLTextAreaElement).value).toBe('沿右侧岩壁前进');
  expect(api.advanceImmersive).toHaveBeenCalledTimes(1);
});
