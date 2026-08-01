from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        self.camera.background_color = "#000000"
        
        # Consolidate title creation to avoid redundant Pango/SVG calls that cause ParseErrors
        if title_text and title_text.strip():
            self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        else:
            self.title = VMobject()
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup layout with specific teaching content
        title = "Prerequisite: The Anatomy of a Sine Wave"
        lines = [
            "Meet Sina the Snake, slithering in a perfect wave.",
            "Slither speed represents the wave's frequency.",
            "Jump height represents the wave's amplitude."
        ]
        self.setup_layout(title, lines)

        # Trackers for wave parameters
        amp_tracker = ValueTracker(0.8)
        freq_tracker = ValueTracker(1.0)
        time_tracker = ValueTracker(0)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")

        # Axes for the snake slither
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": False, "color": GREY_B}
        )
        
        wave_group = VGroup(axes)
        self.place_in_area(wave_group, 'A1', 'C6', scale_factor=0.8)

        # Sine wave plot
        wave = always_redraw(lambda: axes.plot(
            lambda x: amp_tracker.get_value() * np.sin(2 * PI * freq_tracker.get_value() * (x - time_tracker.get_value())),
            color="#00FF00"
        ))

        # Snake Asset
        try:
            snake = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/snake.svg")
        except:
            snake = Triangle().set_color("#00FF00").rotate(-90*DEGREES)
            
        snake.set_color("#00FF00")
        snake.scale(0.3)
        
        def update_snake(mob):
            y_val = amp_tracker.get_value() * np.sin(2 * PI * freq_tracker.get_value() * (0 - time_tracker.get_value()))
            mob.move_to(axes.c2p(0.5, y_val))

        snake.add_updater(update_snake)

        self.play(Create(axes))
        self.play(Create(wave), FadeIn(snake))
        self.add(time_tracker.add_updater(lambda dt: time_tracker.increment_value(dt)))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")

        freq_label = Text("Frequency", font_size=24, color="#FFFF00")
        freq_group = VGroup(freq_label)
        self.place_in_area(freq_group, 'D1', 'E6', scale_factor=0.8)

        self.play(Write(freq_label))
        self.play(freq_tracker.animate.set_value(2.5), run_time=2)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")

        amp_label = Text("Amplitude", font_size=24, color="#FF00FF")
        amp_arrow = DoubleArrow(
            axes.c2p(3.5, -0.8), axes.c2p(3.5, 0.8), 
            color="#FF00FF", buff=0
        )
        
        indicator_group = VGroup(amp_label, amp_arrow).arrange(RIGHT, buff=0.5)
        self.place_in_area(indicator_group, 'D1', 'E6', scale_factor=0.8)

        self.play(FadeOut(freq_label), FadeIn(indicator_group))
        self.play(amp_tracker.animate.set_value(1.4), run_time=2)
        
        fourier_hint = Text("Ready to Decode?", font_size=20, color=BLUE_B)
        self.place_in_area(fourier_hint, 'F4', 'F6', scale_factor=0.6)
        self.play(Write(fourier_hint))
        
        self.wait(3)

        # Cleanup updaters
        time_tracker.clear_updaters()
        snake.clear_updaters()