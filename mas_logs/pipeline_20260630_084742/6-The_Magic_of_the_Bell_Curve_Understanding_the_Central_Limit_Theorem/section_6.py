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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup
        title = "Real-World Power: Why It Matters"
        lines = [
            "We use samples to understand entire giant populations.",
            "Factories test small batches to ensure battery quality.",
            "The bell curve predicts reality from very little data."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_POP = "#9E9E9E"
        COLOR_SAMPLE = "#66BB6A"
        COLOR_ALERT = "#EF5350"
        COLOR_SAFE = "#66BB6A"
        ASSET_BATTERY = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/battery.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_SAMPLE))
        
        # Represent Population - Grey dots
        population_dots = VGroup(*[Dot(radius=0.04, color=COLOR_POP) for _ in range(150)])
        for dot in population_dots:
            dot.move_to(self.grid["C3"] + np.array([
                np.random.uniform(-2, 2),
                np.random.uniform(-1.5, 1.5),
                0
            ]))
        
        pop_label = Text("Population", font_size=18, color=COLOR_POP)
        self.place_at_grid(pop_label, "A3")
        
        # Sample selection box (Spotlight) - Green
        sample_box = Circle(radius=0.8, color=COLOR_SAMPLE, stroke_width=4)
        self.place_at_grid(sample_box, "C3")
        
        # Issue 49: Positioning sample_label at D3 to avoid clutter
        sample_label = Text("Sample", font_size=18, color=COLOR_SAMPLE)
        self.place_at_grid(sample_label, "D3", scale_factor=0.8)

        self.play(FadeIn(population_dots), Write(pop_label))
        self.play(Create(sample_box), Write(sample_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_ALERT)
        )
        self.play(FadeOut(population_dots), FadeOut(pop_label), FadeOut(sample_box), FadeOut(sample_label))

        # Conveyor belt
        belt = Line(self.grid["D1"], self.grid["D6"], color=GREY_E, stroke_width=8)
        
        # Issue 31: Battery icons using asset
        batteries = VGroup(*[SVGMobject(ASSET_BATTERY).scale(0.3).set_color(WHITE) for _ in range(6)])
        for i, b in enumerate(batteries):
            b.move_to(self.grid["D1"] + RIGHT * (i * 0.9))

        self.add(belt)
        self.play(FadeIn(batteries))
        
        # Move batteries along belt
        self.play(batteries.animate.shift(LEFT * 4), run_time=2, rate_func=linear)
        
        # Issue 31 & 47: Alert Flash and Text at A3
        flash = FullScreenRectangle(fill_color=COLOR_ALERT, fill_opacity=0.2, stroke_width=0)
        alert_text = Text("ALERT: DEFECTIVE BATCH!", font_size=24, color=COLOR_ALERT, weight=BOLD)
        self.place_at_grid(alert_text, "A3", scale_factor=1.0)

        self.play(
            FadeIn(flash),
            Write(alert_text),
            run_time=0.2
        )
        self.play(FadeOut(flash), run_time=0.2)
        self.play(FadeIn(flash), run_time=0.2)
        self.play(FadeOut(flash), run_time=0.2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_SAFE)
        )
        self.play(FadeOut(belt), FadeOut(batteries), FadeOut(alert_text))

        # Bell Curve
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.5, 0.1],
            axis_config={"include_tip": False},
            x_length=4,
            y_length=2.5
        )
        self.place_in_area(axes, "B2", "F5")
        
        curve = axes.plot(lambda x: (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * x**2), color=BLUE)
        
        # Safe Zone
        safe_area = axes.get_area(curve, x_range=[-1.5, 1.5], color=COLOR_SAFE, opacity=0.3)
        safe_label = Text("Safe Zone", font_size=14, color=COLOR_SAFE)
        # Issue 48: safe_label at B4
        self.place_at_grid(safe_label, "B4", scale_factor=0.8)

        self.play(Create(axes), Create(curve))
        self.play(FadeIn(safe_area), Write(safe_label))

        # Place single mean dot inside safe zone
        mean_dot = Dot(axes.c2p(0.5, 0), color=WHITE, radius=0.1)
        mean_label = Text("Sample Mean", font_size=16, color=WHITE).next_to(mean_dot, UP, buff=0.1)
        
        self.play(FadeIn(mean_dot), Write(mean_label))
        self.wait(2)
