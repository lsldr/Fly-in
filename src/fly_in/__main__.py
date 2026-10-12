"""Console entry point for the fly-in command-line application."""

from fly_in.cli import CliApplication


class Main:
    """Adapter that forwards console arguments to the CLI application.

    Attributes:
        cli (CliApplication): Application receiving the command arguments.
    """

    def __init__(self, cli: CliApplication | None = None) -> None:
        """Bind the CLI application.

        Args:
            cli (CliApplication | None): Application to run, or ``None`` to
                create the default CLI application.
        """
        self.cli = cli if cli is not None else CliApplication()

    def run(self, argv: list[str] | None = None) -> None:
        """Launch the CLI workflow.

        Args:
            argv (list[str] | None): Arguments excluding the executable name;
                ``None`` uses the process command line.
        """
        self.cli.run(argv)


def main(argv: list[str] | None = None) -> None:
    """Adapt the console command to the application object.

    Args:
        argv (list[str] | None): Arguments excluding the executable name;
            ``None`` uses the process command line.
    """
    Main().run(argv)


if __name__ == "__main__":
    main()
