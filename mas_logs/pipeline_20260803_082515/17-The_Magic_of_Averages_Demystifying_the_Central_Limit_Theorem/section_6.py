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
        # Setup the layout with title and lecture lines
        self.setup_layout(
            "Real-World Application: Why It Matters",
            [
                "The CLT is the powerhouse of modern statistics.",
                "We can study any population using normal distribution math.",
                "This allows for accurate predictions from small samples."
            ]
        )
        
        KHAKI = "#F0E68C"
        CYAN = "#00FFFF"
        GREEN = "#00FF00"
        LIGHT_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(KHAKI)
        
        # Large box representing 1 million bulbs (Fix: Resize box per Issue 37)
        large_box = Rectangle(width=2.0, height=2.0, color=KHAKI, fill_opacity=0.1)
        self.place_in_area(large_box, 'B1', 'C3', scale_factor=0.8)
        
        box_label = Text("1 Million Lightbulbs", font_size=18, color=KHAKI)
        self.place_at_grid(box_label, 'A2')
        
        # Use SVG Asset for lightbulb icon (Issue 22)
        bulb_icon_large = SVGMobject(LIGHT_ASSET).set_color(KHAKI)
        self.place_at_grid(bulb_icon_large, 'B2', scale_factor=0.5)
        
        # Representative population dots
        np.random.seed(42)
        pop_dots = VGroup(*[
            Dot(radius=0.02, color=KHAKI).move_to(
                large_box.get_center() + np.array([np.random.uniform(-0.7, 0.7), np.random.uniform(-0.7, 0.7), 0])
            ) for _ in range(40)
        ])

        self.play(
            Create(large_box),
            Write(box_label),
            FadeIn(bulb_icon_large, shift=UP),
            FadeIn(pop_dots)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(CYAN)
        
        # Create a small cluster using SVG Assets (Issue 22)
        # Fix: Move sample_group to avoid overlap with bell curve (Issue 35)
        sample_group = VGroup(*[
            SVGMobject(LIGHT_ASSET).set_color(KHAKI).scale(0.15)
            for _ in range(12)
        ]).arrange_in_grid(rows=3, cols=4, buff=0.1)
        self.place_in_area(sample_group, 'B4', 'C6', scale_factor=0.6)
        
        # Fix: Better label position (Issue 36)
        sample_label = Text("Sample: 50 Bulbs", font_size=16, color=WHITE)
        self.place_at_grid(sample_label, 'A5')

        # Normal distribution curve (bell curve) in separate area
        ax = Axes(
            x_range=[-2.5, 2.5],
            y_range=[0, 1],
            x_length=2.5,
            y_length=1.5,
            axis_config={"include_tip": False, "include_ticks": False}
        ).set_color(CYAN)
        
        curve = ax.plot(
            lambda x: 0.8 * np.exp(-x**2),
            color=CYAN
        )
        bell_curve_group = VGroup(ax, curve)
        # Place bell curve in bottom area
        self.place_in_area(bell_curve_group, 'E4', 'F6', scale_factor=1.0)
        
        # Animation: Pulling bulbs from the box to the sample area
        self.play(
            AnimationGroup(*[
                ReplacementTransform(pop_dots[i].copy(), sample_group[i % len(sample_group)])
                for i in range(12)
            ], lag_ratio=0.1),
            Write(sample_label)
        )
        self.play(Create(bell_curve_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        
        # Inference arrow points from sample/curve back to population
        inference_arrow = Arrow(
            start=self.grid['D5'],
            end=self.grid['D2'],
            color=GREEN,
            buff=0.2,
            stroke_width=6
        )
        inference_label = Text("Inference", font_size=20, color=GREEN)
        inference_label.next_to(inference_arrow, RIGHT, buff=0.2)
        
        self.play(
            GrowArrow(inference_arrow),
            Write(inference_label)
        )
        self.wait(2)

        # Reset colors for final state
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
