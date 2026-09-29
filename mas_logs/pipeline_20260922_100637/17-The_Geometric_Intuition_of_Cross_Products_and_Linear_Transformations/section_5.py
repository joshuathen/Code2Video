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
        self.setup_layout("Summary and Real-World Application", [
            "Cross products link geometry to linear algebra.", 
            "They compute areas, volumes, and orientations.", 
            "Essential for physics and game rendering."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Cross products link geometry to linear algebra.
        self.lecture[0].set_color(BLUE)
        
        wrench = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wrench.svg").set_color(BLUE)
        self.place_at_grid(wrench, "B3")
        
        formula = MathTex(r"\\mathbf{a} \\times \\mathbf{b} = \\det \\begin{pmatrix} \\mathbf{i} & \\mathbf{j} & \\mathbf{k} \\\\ a_1 & a_2 & a_3 \\\\ b_1 & b_2 & b_3 \\end{pmatrix}").scale(0.7)
        self.place_in_area(formula, 'B1', 'C3', scale_factor=0.9)
        self.play(DrawBorderThenFill(wrench), Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # They compute areas, volumes, and orientations.
        self.lecture[1].set_color(YELLOW)
        
        box = Cube(side_length=1.5, fill_opacity=0.3, stroke_width=2).set_color(YELLOW)
        self.place_at_grid(box, 'D3', scale_factor=0.7)
        self.play(Create(box))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Essential for physics and game rendering.
        self.lecture[2].set_color(GREEN)
        
        engine = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/engine.svg").set_color(GREEN)
        torque_vec = Arrow(start=ORIGIN, end=UP*1.2, color=GREEN)
        torque_label = Text("Torque", font_size=20).next_to(torque_vec, UP, buff=0.1)
        torque_group = VGroup(engine, torque_vec, torque_label)
        self.place_at_grid(torque_group, 'D5', scale_factor=0.8)
        self.play(FadeIn(engine), GrowArrow(torque_vec), Write(torque_label))
        self.wait(2)
