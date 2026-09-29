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

class Section2Scene(ThreeDScene, TeachingScene):
    def construct(self):
        self.setup_layout("The 3D Definition: Creating Space", [
            "3D cross product creates a new perpendicular vector.",
            "Use the Right-Hand Rule to find direction.",
            "Curl fingers from A to B; thumb points result."
        ])
        
        # --- Create 3D coordinate space ---
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2])
        vec_a = Arrow3D(start=ORIGIN, end=[1, 0, 0], color="#FF6666")
        vec_b = Arrow3D(start=ORIGIN, end=[0, 1, 0], color="#66CCFF")
        vec_c = Arrow3D(start=ORIGIN, end=[0, 0, 1], color="#66FF66")
        
        # Load asset
        hand_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hand.svg")
        
        scene_group = VGroup(axes, vec_a, vec_b, vec_c)
        # Apply updated positioning as requested
        self.place_in_area(scene_group, 'A4', 'C6', scale_factor=0.55)
        
        # Initial render
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        self.play(Create(axes), Create(vec_a), Create(vec_b), Create(vec_c))
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # --- Place hand icon for RHR context ---
        # Apply updated positioning as requested
        self.place_at_grid(hand_icon, 'F6', scale_factor=0.4)
        self.add_fixed_in_frame_mobjects(hand_icon)
        self.play(FadeIn(hand_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # --- Show 3D parallelepiped ---
        box = VGroup(
            Line(vec_a.get_end(), [1, 1, 0]), Line(vec_b.get_end(), [1, 1, 0]),
            Line(vec_a.get_end(), [1, 0, 1]), Line(vec_c.get_end(), [1, 0, 1]),
            Line(vec_b.get_end(), [0, 1, 1]), Line(vec_c.get_end(), [0, 1, 1]),
            Line([1, 1, 0], [1, 1, 1]), Line([1, 0, 1], [1, 1, 1]), Line([0, 1, 1], [1, 1, 1])
        ).set_color(WHITE)
        self.play(Create(box))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Rotate view
        self.move_camera(theta=135 * DEGREES, run_time=2)
        
        self.wait(2)
