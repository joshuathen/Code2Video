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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Shrink the interval between these two points.",
            "The secant slides towards a single point.",
            "It transforms into the tangent line.",
            "Imagine a camera shutter speed becoming faster.",
            "The 'blur' reveals the exact instant direction."
        ]
        self.setup_layout("The Concept of Transformation: Secant to Tangent", lecture_lines)
        
        # Setup Plot Area
        axes = Axes(x_range=[0, 3], y_range=[0, 3], axis_config={"include_tip": False})
        self.place_in_area(axes, 'B3', 'F5', scale_factor=0.55)
        
        # Labels and Icons
        x_label = MathTex("x")
        self.place_at_grid(x_label, 'F4', scale_factor=0.6)
        
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        self.place_at_grid(camera_icon, 'A4', scale_factor=0.5)
        
        curve = axes.plot(lambda x: x**2 / 3, color=BLUE)
        
        # Points
        A = axes.c2p(1, 1/3)
        B_start = axes.c2p(2.5, 2.5**2 / 3)
        
        point_a = Dot(A, color=WHITE)
        point_b = Dot(B_start, color=YELLOW)
        
        # Secant Line
        secant = Line(point_a.get_center(), point_b.get_center(), color=YELLOW)
        
        # Add to scene
        self.add(axes, curve, point_a, point_b, secant, x_label, camera_icon)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # B moves to A
        def update_secant(mob):
            mob.put_start_and_end_on(point_a.get_center(), point_b.get_center())
            
        secant.add_updater(update_secant)
        self.play(point_b.animate.move_to(axes.c2p(1.2, 1.2**2/3)), run_time=2)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        # Final transition to tangent
        tangent = TangentLine(curve, alpha=1/3, length=2, color=GREEN)
        self.play(ReplacementTransform(secant, tangent), run_time=1)
        secant.remove_updater(update_secant)
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(BLUE))
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(GREEN))
        self.play(Indicate(tangent))
        self.wait(2)
