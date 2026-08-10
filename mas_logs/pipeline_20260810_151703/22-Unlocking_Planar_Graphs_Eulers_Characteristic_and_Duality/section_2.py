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
        lecture_lines = [
            "Euler's formula links graph components.",
            "V minus E plus F equals 2.",
            "Moving vertices preserves this constant value."
        ]
        self.setup_layout("Euler’s Characteristic Formula (V - E + F = 2)", lecture_lines)
        
        # Assets
        polyhedron = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/polyhedron.svg")
        polygon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/polygon.svg")
        
        # Formula setup
        formula = MathTex(r"V", r"-", r"E", r"+", r"F", r"=", r"2")
        self.place_in_area(formula, 'C3', 'D4', scale_factor=1.2)
        
        v_part = formula[0]
        e_part = formula[2]
        f_part = formula[4]
        res_part = formula[6]
        
        # Labels
        lbl_v = Text("Vertices", font_size=20, color="#FF5733")
        lbl_e = Text("Edges", font_size=20, color="#33FF57")
        lbl_f = Text("Faces", font_size=20, color="#3357FF")
        
        self.place_at_grid(lbl_v, 'B3', scale_factor=0.8)
        self.place_at_grid(lbl_e, 'D3', scale_factor=0.8)
        self.place_at_grid(lbl_f, 'B5', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(polyhedron, 'A4', scale_factor=0.5)
        self.play(FadeIn(formula), FadeIn(polyhedron))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(
            v_part.animate.set_color("#FF5733"),
            e_part.animate.set_color("#33FF57"),
            f_part.animate.set_color("#3357FF"),
            FadeIn(lbl_v), FadeIn(lbl_e), FadeIn(lbl_f)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.place_at_grid(polygon, 'F4', scale_factor=0.4)
        self.play(FadeIn(polygon))
        self.play(res_part.animate.set_color(WHITE).set_opacity(0.5))
        self.play(Flash(res_part, color=WHITE, line_length=0.2, num_lines=15))
        self.wait(2)
