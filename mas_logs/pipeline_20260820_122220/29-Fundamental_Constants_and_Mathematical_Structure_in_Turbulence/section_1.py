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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Turbulence Puzzle: Defining the Chaos", [
            "Turbulence is chaotic, unpredictable fluid motion.",
            "The Reynolds Number governs this complexity.",
            "It compares inertial to viscous forces.",
            "High Re creates chaotic, swirling eddies.",
            "Finley the trout navigates these flows."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Turbulence is chaotic, unpredictable fluid motion.
        self.lecture[0].set_color("#FFFFFF")
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/trout.svg
        chaos_cloud = VGroup(*[Dot(radius=0.05, color="#FF4500").move_to(self.grid[f"{row}{col}"] + np.random.normal(0, 0.2, 3)) 
                             for row in "BCDE" for col in "3456"])
        self.play(FadeIn(chaos_cloud))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The Reynolds Number governs this complexity.
        self.lecture[0].set_color("#888888")
        self.lecture[1].set_color("#FFD700")
        re_label = MathTex(r"Re = \frac{uL}{\nu}", color="#FFD700")
        self.place_in_area(re_label, 'A4', 'B6', scale_factor=0.9)
        self.play(Write(re_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # It compares inertial to viscous forces.
        self.lecture[1].set_color("#888888")
        self.lecture[2].set_color("#00FF00")
        inertia = Text("Inertial", color="#FF0000", font_size=24)
        viscous = Text("Viscous", color="#0000FF", font_size=24)
        self.place_at_grid(inertia, 'C2', scale_factor=1.0)
        self.place_at_grid(viscous, 'C5', scale_factor=1.0)
        self.play(FadeIn(inertia), FadeIn(viscous))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # High Re creates chaotic, swirling eddies.
        self.lecture[2].set_color("#888888")
        self.lecture[3].set_color("#FF4500")
        eddy = Arc(radius=0.5, start_angle=0, angle=2*PI, color="#FF4500")
        self.place_at_grid(eddy, 'E3', scale_factor=0.8)
        self.play(Create(eddy), Rotate(eddy, angle=PI))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Finley the trout navigates these flows.
        self.lecture[3].set_color("#888888")
        self.lecture[4].set_color("#00FFFF")
        # Final scene showing full cascade in #00FFFF. [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/trout.svg]
        finley = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/trout.svg").set_color("#00FFFF")
        self.place_at_grid(finley, "E5", scale_factor=0.5)
        self.play(FadeIn(finley))
        self.play(finley.animate.shift(LEFT * 2), run_time=2)
        self.wait(2)
