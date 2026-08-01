from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
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

class Section4Scene(TeachingScene):
    def construct(self):
        # Data
        title_text = "Exotic Examples: Functions as Vectors"
        lecture_lines = [
            "Functions can be treated as vectors in abstract spaces.",
            "Adding two functions creates a new, combined curve.",
            "Scaling a function changes its height or amplitude.",
            "Like arrows, functions satisfy all eight vector axioms.",
            "This transforms calculus into a branch of linear algebra."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        PINK_CLR = "#FF69B4"
        CYAN_CLR = "#00FFFF"
        PURPLE_CLR = "#8A2BE2"
        HIGHLIGHT_CLR = "#FFFF00"
        
        # Assets
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sine.svg]
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wave.svg]
        sine_icon_pink = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sine.svg", color=PINK_CLR)
        sine_icon_cyan = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sine.svg", color=CYAN_CLR)
        wave_icon_purple = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wave.svg", color=PURPLE_CLR)
        
        # Elements - Axes for plotting functions
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": False, "color": GREY_C},
            tips=False
        )
        self.place_in_area(axes, "A1", "F6")
        
        # Scalar tracker for the pink wave amplitude
        amp_p = ValueTracker(0.5)
        
        # Pink wave: f(x) = amp * sin(2x)
        # Using always_redraw for amplitude animation
        pink_wave = always_redraw(lambda: axes.plot(
            lambda x: amp_p.get_value() * np.sin(2 * x),
            color=PINK_CLR
        ))
        
        # Cyan wave: g(x) = 0.5 * sin(3x) (static)
        cyan_wave = axes.plot(
            lambda x: 0.5 * np.sin(3 * x),
            color=CYAN_CLR
        )
        
        # Purple wave: h(x) = f(x) + g(x)
        # Using always_redraw to reflect changes in pink_wave amplitude
        purple_wave = always_redraw(lambda: axes.plot(
            lambda x: amp_p.get_value() * np.sin(2 * x) + 0.5 * np.sin(3 * x),
            color=PURPLE_CLR
        ))

        # Labels for the curves
        pink_label = Text("f(x)", color=PINK_CLR, font_size=20)
        cyan_label = Text("g(x)", color=CYAN_CLR, font_size=20)
        purple_label = Text("f(x) + g(x)", color=PURPLE_CLR, font_size=20)
        
        # Positioning labels and icons according to VideoCritic feedback
        # Issue 27: Move f(x) label to row B for margin
        self.place_at_grid(pink_label, "B5", scale_factor=1.0)
        self.place_at_grid(sine_icon_pink, "B4", scale_factor=0.4)
        
        # Issue 28: Move g(x) label to B6 for balance
        self.place_at_grid(cyan_label, "B6", scale_factor=1.0)
        self.place_at_grid(sine_icon_cyan, "A6", scale_factor=0.4)
        
        # Issue 26: Use area for long purple label to avoid overlap
        self.place_in_area(purple_label, "F5", "F6", scale_factor=0.8)
        self.place_at_grid(wave_icon_purple, "F4", scale_factor=0.4)

        # === Animation for Lecture Line 1 ===
        # "Functions can be treated as vectors in abstract spaces."
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_CLR))
        self.play(Create(axes))
        self.play(
            Create(pink_wave), 
            FadeIn(sine_icon_pink), 
            Write(pink_label),
            Create(cyan_wave),
            FadeIn(sine_icon_cyan),
            Write(cyan_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # "Adding two functions creates a new, combined curve."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(PURPLE_CLR)
        )
        self.play(
            Create(purple_wave), 
            FadeIn(wave_icon_purple), 
            Write(purple_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # "Scaling a function changes its height or amplitude."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(PINK_CLR)
        )
        # Animate the amplitude change (scalar multiplication)
        self.play(amp_p.animate.set_value(1.2), run_time=2)
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # "Like arrows, functions satisfy all eight vector axioms."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(HIGHLIGHT_CLR)
        )
        self.wait(3)

        # === Animation for Lecture Line 5 ===
        # "This transforms calculus into a branch of linear algebra."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT_CLR)
        )
        self.wait(3)
