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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Golden Triangle of Notation", [
            "Exponents, roots, logs are linked.",
            "Three ways to read growth.",
            "Three squared is nine.",
            "Root nine is three.",
            "Log base three nine is two."
        ])
        
        # Triangle nodes
        node_exp = MathTex("3^2=9", color="#FF9999")
        node_root = MathTex(r"\sqrt{9}=3", color="#99FF99")
        node_log = MathTex(r"\log_3{9}=2", color="#9999FF")
        
        # Applying requested layout fixes (VideoCritic/Orchestrator)
        self.place_in_area(node_exp, 'A2', 'B3', scale_factor=0.9)
        self.place_at_grid(node_root, 'D2', scale_factor=1.0)
        self.place_at_grid(node_log, 'D5', scale_factor=1.0)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg", color=WHITE)
        self.place_at_grid(calculator, 'C3', scale_factor=0.5)
        
        # Triangle connectors (need positions relative to nodes)
        line1 = Line(node_exp.get_bottom(), node_root.get_top(), color=WHITE)
        line2 = Line(node_root.get_right(), node_log.get_left(), color=WHITE)
        line3 = Line(node_log.get_top(), node_exp.get_bottom(), color=WHITE)
        
        triangle = VGroup(line1, line2, line3)
        label = Text("The Golden Trio", font_size=24, color=YELLOW)
        self.place_at_grid(label, 'E4', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(node_exp), FadeIn(node_root), FadeIn(node_log), FadeIn(calculator), self.lecture[0].animate.set_color("#FF9999"))
        
        # === Animation for Lecture Line 2 ===
        self.play(Create(triangle), self.lecture[1].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 3 ===
        self.play(Indicate(node_exp), self.lecture[2].animate.set_color("#FF9999"))
        
        # === Animation for Lecture Line 4 ===
        self.play(Indicate(node_root), self.lecture[3].animate.set_color("#99FF99"))
        
        # === Animation for Lecture Line 5 ===
        self.play(Indicate(node_log), FadeIn(label), self.lecture[4].animate.set_color("#9999FF"))
        
        self.wait(2)
