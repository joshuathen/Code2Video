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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Vector Addition: The Tip-to-Tail Rule", [
            "Vector addition combines two vectors.",
            "Place second tail at first tip.",
            "Resultant is the path's total displacement."
        ])
        
        # Define vectors
        origin = self.grid['E2']
        vec_a_end = origin + np.array([1.2, 0.6, 0])
        vec_b_end = vec_a_end + np.array([0.6, 1.8, 0])
        
        vec_a = Arrow(origin, vec_a_end, color="#2ECC71", buff=0)
        vec_b = Arrow(vec_a_end, vec_b_end, color="#F1C40F", buff=0)
        res_vec = Arrow(origin, vec_b_end, color="#FFFFFF", buff=0)
        
        label_a = MathTex(r"\\vec{a}", color="#2ECC71").scale(0.7).next_to(vec_a.get_center(), UP, buff=0.1)
        label_b = MathTex(r"\\vec{b}", color="#F1C40F").scale(0.7).next_to(vec_b.get_center(), RIGHT, buff=0.1)
        resultant_formula = MathTex(r"\\vec{a}+\\vec{b}", color="#FFFFFF").scale(0.7)
        self.place_in_area(resultant_formula, 'E4', 'F6', scale_factor=0.8)
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg").scale(0.3)
        self.place_at_grid(compass, 'A4', scale_factor=0.8)
        
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg").scale(0.3)
        self.place_at_grid(ruler, 'B4', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#2ECC71"))
        self.play(Create(vec_a), Write(label_a), FadeIn(compass))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#F1C40F"))
        self.play(Create(vec_b), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.play(Create(res_vec), FadeIn(ruler), Write(resultant_formula))
        self.wait(2)
