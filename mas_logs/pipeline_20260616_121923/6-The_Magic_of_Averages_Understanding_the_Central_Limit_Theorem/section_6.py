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
        lecture_lines = [
            'We moved from messy data to orderly curves.', 
            'The CLT is the bridge to powerful statistics.', 
            'Understand the average to master the whole.'
        ]
        self.setup_layout("Summary & Conclusion", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        
        # Create a cloud of chaotic dots in area A1 to B6
        dots = VGroup()
        colors = ["#ADD8E6", "#FFA500"]
        for _ in range(50):
            dot = Dot(radius=0.05, color=np.random.choice(colors))
            # Random position within top area (A1 to B6)
            x_rand = np.random.uniform(0.5, 5.5)
            y_rand = np.random.uniform(1.2, 2.2)
            dot.move_to([x_rand, y_rand, 0])
            dots.add(dot)
            
        # Funnel Asset Integration [Issue 38]
        funnel_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/funnel.svg")
        funnel_asset.set_color(WHITE)
        # Place funnel in middle area
        self.place_in_area(funnel_asset, "D1", "D6", scale_factor=0.7)
        
        self.play(FadeIn(dots), FadeIn(funnel_asset))
        
        # Funneling animation: dots converge to the funnel center
        funnel_center = funnel_asset.get_center()
        self.play(
            dots.animate.scale(0.1).move_to(funnel_center),
            run_time=2,
            rate_func=exponential_decay
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFD700")
        )
        
        # Emerging Bell Curve in area E1 to F6
        axes = Axes(
            x_range=[-3, 3], y_range=[0, 0.5],
            x_length=4, y_length=2,
            axis_config={"include_tip": False, "include_ticks": False, "color": GREY_C}
        )
        curve = axes.plot(lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color="#FFD700")
        bell_curve_group = VGroup(axes, curve)
        # Fix: Reduced scale to 0.8 and placed in E1-F6 [Issue 54]
        self.place_in_area(bell_curve_group, "E1", "F6", scale_factor=0.8)
        
        # Glowing effect
        glow = curve.copy().set_stroke(width=8, opacity=0.3)
        bell_curve_glow = VGroup(bell_curve_group, glow)
        
        self.play(
            FadeIn(bell_curve_glow, shift=DOWN),
            dots.animate.set_opacity(0),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFFFF")
        )
        
        # Final concluding text
        conclusion_text = Text("The Bridge to Statistical Power", font_size=24, color="#FFFFFF")
        # Fix: Move to C1-C6 and scale to 0.8 to avoid overlap [Issue 53]
        self.place_in_area(conclusion_text, "C1", "C6", scale_factor=0.8)
        
        self.play(Write(conclusion_text))
        self.wait(3)
