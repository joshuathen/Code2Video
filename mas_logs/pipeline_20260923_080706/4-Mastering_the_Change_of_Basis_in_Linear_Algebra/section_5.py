from manim import *

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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Real-World Synthesis & Wrap-up", [
            "Changing basis simplifies complex problems.",
            "Symmetry aligns with your calculation axis.",
            "Diagonal matrices make physics easy."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show summary graphic with [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/molecule.svg]
        molecule = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/molecule.svg")
        molecule.set_color("#3498DB")
        self.place_at_grid(molecule, 'B4', scale_factor=0.7)
        self.play(FadeIn(molecule), run_time=1)
        self.play(self.lecture[0].animate.set_color("#3498DB"), run_time=0.5)

        # === Animation for Lecture Line 2 ===
        # Animate transition from abstract (original basis) to concrete (aligned with symmetry)
        rot_angle = PI/4
        molecule_rotated = molecule.copy().rotate(rot_angle, about_point=molecule.get_center())
        molecule_rotated.set_color(GREEN)
        
        self.play(ReplacementTransform(molecule.copy(), molecule_rotated), run_time=1.5)
        self.play(self.lecture[1].animate.set_color("#3498DB"), run_time=0.5)

        # === Animation for Lecture Line 3 ===
        # Display final concluding text with [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg]
        bridge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        bridge.set_color("#F1C40F")
        diag_matrix = Matrix([[3, 0], [0, 5]], h_buff=0.5)
        
        group_all = VGroup(bridge, diag_matrix).arrange(DOWN)
        
        self.place_at_grid(diag_matrix, 'E5', scale_factor=0.5)
        self.place_in_area(group_all, 'B4', 'F6', scale_factor=0.6)

        self.play(FadeIn(bridge), FadeIn(diag_matrix), run_time=1)
        self.play(self.lecture[2].animate.set_color("#F1C40F"), run_time=0.5)
        self.wait(2)
