import beaker
import beaker.application
import pyteal as pt


gamemanager = beaker.application.Application("gamemanager")


if __name__ == "__main__":
    gamemanager_spec = gamemanager.build()
    print(gamemanager_spec.to_json())
