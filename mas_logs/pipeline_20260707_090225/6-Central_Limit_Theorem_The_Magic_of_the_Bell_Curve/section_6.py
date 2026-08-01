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
        # Colors
        COLOR_1 = "#33CCFF" # Light Blue for Factory/Batteries
        COLOR_2 = "#FFFF00" # Yellow for Bell Curve
        COLOR_3 = "#FFFFFF" # White for Conclusion

        self.setup_layout(
            "Why It Matters: The Statistician's Superpower", 
            [
                "This theorem is a superpower for predicting population behavior.", 
                "Factories use it to guarantee product quality and safety.", 
                "We find order in complexity through the bell curve."
            ]
        )

        # Assets
        FACTORY_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/factory.svg"
        BATTERY_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/battery.svg"

        # === Animation for Lecture Line 1 ===
        # Factory setting: Conveyor belt and batteries
        self.lecture[0].set_color(COLOR_1)
        
        # Use factory asset (Issue 29)
        factory = SVGMobject(FACTORY_ASSET).set_color(COLOR_1)
        self.place_at_grid(factory, "B6", scale_factor=0.6)
        
        belt = Line(self.grid["E1"], self.grid["E6"], color=GREY_B, stroke_width=6)
        
        # Use battery asset (Issue 29)
        batteries = VGroup(*[SVGMobject(BATTERY_ASSET).set_color(COLOR_1) for _ in range(6)])
        for i, battery in enumerate(batteries):
            self.place_at_grid(battery, f"E{i+1}", scale_factor=0.3)
            battery.shift(UP * 0.3)

        self.play(FadeIn(factory), Create(belt))
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT) for b in batteries], lag_ratio=0.2))
        
        # Battery movement loop simulation
        self.play(
            *[b.animate.shift(RIGHT * 1.0) for b in batteries[:-1]],
            batteries[-1].animate.set_opacity(0),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Bell Curve over small sample
        self.lecture[1].set_color(COLOR_2)
        
        # Highlight sample (center batteries) using battery assets
        sample_rect = SurroundingRectangle(VGroup(batteries[2], batteries[3]), color=COLOR_2, buff=0.1)
        
        # Bell Curve
        axes = Axes(
            x_range=[-3, 3], 
            y_range=[0, 1], 
            x_length=4, 
            y_length=2.5, 
            axis_config={"include_tip": False, "color": GREY_D}
        )
        curve = axes.plot(lambda x: np.exp(-x**2), color=COLOR_2)
        curve_label = Text("Normal Distribution", font_size=18, color=COLOR_2)
        curve_group = VGroup(axes, curve, curve_label).arrange(DOWN, buff=0.2)
        
        # Resolve Issue 46 & 52: Position curve_group to optimize mid-section space
        self.place_in_area(curve_group, "A2", "D5", scale_factor=0.85)

        self.play(Create(sample_rect))
        self.play(FadeIn(curve_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final conclusion text
        self.lecture[2].set_color(COLOR_3)
        
        conclusion_text = Text("Predicting the Unknown", font_size=36, color=COLOR_3, weight=BOLD)
        
        # Resolve Issue 45 & 52: Position conclusion_text at the bottom to avoid overlapping curve
        self.place_in_area(conclusion_text, "E2", "F5", scale_factor=0.7)
        
        # Transition to conclusion: clean up factory assets while keeping curve visible
        self.play(
            FadeOut(factory),
            FadeOut(belt),
            FadeOut(batteries),
            FadeOut(sample_rect),
            FadeIn(conclusion_text)
        )
        
        self.play(Indicate(conclusion_text, scale_factor=1.1, color=COLOR_3))
        self.wait(3)

        # Fade out all
        self.play(
            FadeOut(self.lecture), 
            FadeOut(self.title), 
            FadeOut(conclusion_text),
            FadeOut(curve_group)
        )
