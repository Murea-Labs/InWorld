import { useState, type FormEvent } from 'react'
import type { User } from '../types'
import Dialog from '../dialogs/Dialog'
import { t } from '../i18n'

interface Props {
  users: User[]
  loading: boolean
  onSelect: (user: User, password?: string) => Promise<void>
  onCreate: (name: string, password?: string) => Promise<void>
  onDelete: (user: User, password?: string) => Promise<void>
}

export default function UserGate({ users, loading, onSelect, onCreate, onDelete }: Props) {
  const [name, setName] = useState('')
  const [pw, setPw] = useState('')
  // Clicked a user with a password: expand the password field and wait for it.
  const [asking, setAsking] = useState<User | null>(null)
  const [askPw, setAskPw] = useState('')
  const [deleting, setDeleting] = useState<User | null>(null)
  const [deletePw, setDeletePw] = useState('')
  const [working, setWorking] = useState('')
  const [error, setError] = useState('')

  const choose = async (user: User, password = '') => {
    // For a user with a password the first click only expands the field and sends nothing;
    // sending right away always returns 401 and shows an error before the field.
    if (user.locked && !password) {
      setError('')
      setAskPw('')
      setAsking(user)
      return
    }
    setError('')
    setWorking(user.id)
    try {
      await onSelect(user, password)
    } catch (e) {
      setError((e as Error).message)
      setWorking('')
    }
  }

  const create = async (event: FormEvent) => {
    event.preventDefault()
    const value = name.trim()
    if (!value) return
    setError('')
    setWorking('new')
    try {
      await onCreate(value, pw)
    } catch (e) {
      setError((e as Error).message)
      setWorking('')
    }
  }

  const remove = async () => {
    if (!deleting) return
    setError('')
    setWorking(`delete:${deleting.id}`)
    try {
      await onDelete(deleting, deletePw)
      setDeleting(null)
      setDeletePw('')
      setWorking('')
    } catch (e) {
      setError((e as Error).message)
      setWorking('')
    }
  }

  return (
    <main className="user-gate">
      <div className="user-gate-glow" aria-hidden="true" />
      <section className="user-gate-card" aria-labelledby="user-gate-title">
        <div className="user-gate-brand">
          <span className="user-gate-mark" aria-hidden="true">A</span>
          <div>
            <p>Murea-Mind</p>
            <span>{t('进入你的世界')}</span>
          </div>
        </div>

        <div className="user-gate-copy">
          <p className="eyebrow">Choose your profile</p>
          <h1 id="user-gate-title">{t('这次由谁来探索？')}</h1>
          <p>{t('选择一个已有用户，或用新名字开始。每个人只会看到自己创建的世界。')}</p>
        </div>

        <div className="user-gate-users" aria-busy={loading}>
          {loading ? (
            <div className="user-gate-loading"><i />{t('正在读取用户…')}</div>
          ) : users.map((user) => (
            <div key={user.id} className="user-profile-row">
              <div className={'user-profile-actions' + (deleting?.id === user.id ? ' deleting' : '')}>
                <button
                  type="button"
                  className={'user-profile' + (asking?.id === user.id ? ' asking' : '')}
                  disabled={!!working}
                  onClick={() => {
                    setDeleting(null)
                    void choose(user)
                  }}
                >
                  <span className="user-profile-avatar">{user.name.slice(0, 1).toUpperCase()}</span>
                  <span className="user-profile-copy">
                    <b>{user.name}</b>
                    <small>
                      {user.id === 'u-default' ? t('默认用户')
                        : user.locked ? t('需要密码') : t('进入个人世界')}
                    </small>
                  </span>
                  {user.locked && (
                    <svg className="user-profile-lock" viewBox="0 0 16 16" aria-hidden="true">
                      <rect x="3.2" y="7" width="9.6" height="7" rx="1.6" />
                      <path d="M5.6 7V5.2a2.4 2.4 0 0 1 4.8 0V7" />
                    </svg>
                  )}
                  <svg viewBox="0 0 20 20" aria-hidden="true">
                    <path d="m7 4 6 6-6 6" />
                  </svg>
                </button>
                {user.id !== 'u-default' && (
                  <button
                    type="button"
                    className="user-profile-delete"
                    disabled={!!working}
                    aria-label={t('注销用户 {name}', { name: user.name })}
                    title={t('注销用户 {name}', { name: user.name })}
                    onClick={() => {
                      setError('')
                      setAsking(null)
                      setDeletePw('')
                      setDeleting((current) => current?.id === user.id ? null : user)
                    }}
                  >
                    {t('注销')}
                  </button>
                )}
              </div>
              {asking?.id === user.id && (
                <form
                  className="user-unlock"
                  onSubmit={(event) => {
                    event.preventDefault()
                    void choose(user, askPw)
                  }}
                >
                  <input
                    type="password"
                    value={askPw}
                    autoFocus
                    disabled={!!working}
                    autoComplete="current-password"
                    placeholder={t('{name} 的密码', { name: user.name })}
                    onChange={(event) => setAskPw(event.target.value)}
                  />
                  <button className="primary" disabled={!askPw || !!working}>
                    {working === user.id ? '…' : t('进入')}
                  </button>
                </form>
              )}
            </div>
          ))}
        </div>

        <div className="user-gate-divider"><span>{t('注册新用户')}</span></div>

        <form className="user-create" onSubmit={(event) => void create(event)}>
          <label htmlFor="new-user-name">{t('用户名')}</label>
          <div>
            <input
              id="new-user-name"
              value={name}
              maxLength={40}
              disabled={!!working}
              autoComplete="off"
              placeholder={t('输入一个名字')}
              onChange={(event) => setName(event.target.value)}
            />
            <button className="primary" disabled={!name.trim() || !!working}>
              {working === 'new' ? t('创建中…') : t('创建并进入')}
            </button>
          </div>
          <label htmlFor="new-user-pw">{t('密码（选填）')}</label>
          <div>
            <input
              id="new-user-pw"
              type="password"
              value={pw}
              disabled={!!working}
              autoComplete="new-password"
              placeholder={t('留空就不设密码，至少 6 位')}
              onChange={(event) => setPw(event.target.value)}
            />
          </div>
          {/* Say clearly what having no password means: "may be empty" is not "empty is fine". */}
          <p>{t('设了密码，别人就进不来你的世界；留空的话，谁点你的名字都能进。')}</p>
        </form>

        {error && <div className="user-gate-error" role="alert">{error}</div>}
      </section>
      <Dialog
        open={!!deleting}
        className="user-delete-dialog"
        title={deleting ? t('确认注销「{name}」', { name: deleting.name }) : t('确认注销')}
        sub={t('注销后，这个用户将从本机移除。')}
        busy={working === `delete:${deleting?.id}`}
        okLabel={t('确认注销')}
        okClassName="danger"
        okDisabled={!!deleting?.locked && !deletePw}
        onClose={() => {
          if (working) return
          setDeleting(null)
          setDeletePw('')
        }}
        onOk={() => void remove()}
        >
        <div className="user-delete-modal-copy">
          <p>{t('这个用户创建的个人世界及其专属素材也会一并删除，操作无法恢复。')}</p>
        </div>
        {deleting?.locked && (
          <input
            type="password"
            value={deletePw}
            autoFocus
            disabled={!!working}
            autoComplete="current-password"
            placeholder={t('{name} 的密码', { name: deleting.name })}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && deletePw) void remove()
            }}
            onChange={(event) => setDeletePw(event.target.value)}
          />
        )}
      </Dialog>
    </main>
  )
}
