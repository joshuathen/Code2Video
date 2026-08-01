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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite Check: What is a Basis?", [
            "A basis is a set of independent spanning vectors.",
            "The standard basis uses unit vectors i and j.",
            "This creates a familiar, rigid square grid.",
            "Vector v is just instructions relative to this grid.",
            "Change the basis, and you change the instructions."
        ])
        
        # Asset Loading (Issue 23)
        grid_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        instr_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/instructions.svg")

        # --- PREPARE SYSTEM ---
        # Coordinate System Group
        origin = ORIGIN
        # Scale unit vectors so 2i + 3j fits in the 3x3 area
        u_length = 0.4
        i_vec = Arrow(start=origin, end=RIGHT * u_length, color="#FF0000", buff=0)
        j_vec = Arrow(start=origin, end=UP * u_length, color="#00FF00", buff=0)
        i_label = MathTex("\\hat{i}", color="#FF0000", font_size=20).next_to(RIGHT * u_length, DOWN, buff=0.1)
        j_label = MathTex("\\hat{j}", color="#00FF00", font_size=20).next_to(UP * u_length, LEFT, buff=0.1)
        
        grid_svg.set_color("#444444")
        # Ensure grid_svg is scaled to match the 3x3 area roughly
        grid_svg.width = 3.0
        
        grid_and_vectors = VGroup(grid_svg, i_vec, j_vec, i_label, j_label)
        # Issue 28: Fix grid and vector system layout.
        self.place_in_area(grid_and_vectors, 'B3', 'E6', scale_factor=0.8)
        
        origin_pt = i_vec.get_start()
        u_i = i_vec.get_end() - origin_pt
        u_j = j_vec.get_end() - origin_pt

        # --- ANIMATIONS ---

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        v1 = Arrow(start=origin_pt, end=origin_pt + 1.1*u_i + 0.4*u_j, color="#00FFFF", buff=0)
        v2 = Arrow(start=origin_pt, end=origin_pt - 0.2*u_i + 0.9*u_j, color="#00FFFF", buff=0)
        basis_b_label = Text("Basis B", font_size=20, color="#00FFFF")
        # Issue 29: Place 'Basis B' label at 'B4'
        self.place_at_grid(basis_b_label, 'B4', scale_factor=0.8)
        
        self.play(Create(v1), Create(v2), Write(basis_b_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF0000")
        self.play(
            FadeOut(v1), FadeOut(v2), FadeOut(basis_b_label),
            Create(i_vec), Create(j_vec), Write(i_label), Write(j_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#888888")
        # Use Asset: grid.svg for the grid (Issue 23)
        self.play(FadeIn(grid_svg))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFFFFF")
        
        # Show vector v = 2i + 3j
        v_end = origin_pt + 2*u_i + 3*u_j 
        v_vec = Arrow(start=origin_pt, end=v_end, color="#FFFFFF", buff=0)
        v_label = MathTex("v = [2, 3]^T", color="#FFFFFF", font_size=22)
        # Issue 30: place_at_grid(v_label, 'B6', scale_factor=0.7)
        self.place_at_grid(v_label, 'B6', scale_factor=0.7)
        
        step_h = DashedLine(start=origin_pt, end=origin_pt + 2*u_i, color="#FFFFFF", stroke_width=2)
        step_v = DashedLine(start=origin_pt + 2*u_i, end=v_end, color="#FFFFFF", stroke_width=2)
        
        self.play(Create(step_h))
        self.play(Create(step_v))
        self.play(Create(v_vec), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFF00")
        
        # Use Asset: instructions.svg (Issue 23)
        instr_svg.set_color("#FFFF00")
        self.place_at_grid(instr_svg, 'E3', scale_factor=0.6)
        
        # Calculate new basis vector rotations
        from manim.utils.space_ops import rotate_vector
        new_u_i = rotate_vector(u_i, 30*DEGREES) * 1.2
        new_u_j = rotate_vector(u_j, -15*DEGREES) * 0.8
        
        self.play(
            i_vec.animate.put_start_and_end_on(origin_pt, origin_pt + new_u_i),
            j_vec.animate.put_start_and_end_on(origin_pt, origin_pt + new_u_j),
            i_label.animate.next_to(origin_pt + new_u_i, RIGHT, buff=0.1),
            j_label.animate.next_to(origin_pt + new_u_j, UP, buff=0.1),
            grid_svg.animate.rotate(20*DEGREES).scale(1.1).set_opacity(0.3),
            FadeOut(step_h), FadeOut(step_v),
            v_vec.animate.set_opacity(0.3),
            v_label.animate.set_opacity(0.3),
            FadeIn(instr_svg)
        )
        self.wait(2)
