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

class Section1Scene(TeachingScene):
    def construct(self):
        # Fetching data from storyboard
        title_text = "The Hook: From Scalar Growth to System Growth"
        lecture_lines = [
            "Scalar growth follows the exponential function e to the at.",
            "Systems of equations model interacting real-world objects.",
            "Can we raise e to a matrix power instead?"
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        emerald_green = "#50C878"
        gold = "#FFD700"
        white = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Use MarkupText to simulate math formatting without LaTeX dependency
        self.play(self.lecture[0].animate.set_color(emerald_green))
        
        scalar_growth = MarkupText('y(t) = e<sup>at</sup>', color=emerald_green)
        # Fix Issue 22: Positioning
        self.place_in_area(scalar_growth, 'B4', 'C5', scale_factor=1.5)
        
        self.play(Write(scalar_growth))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Manual construction of vector/matrix to avoid Matrix/MathTex LaTeX errors
        self.play(self.lecture[1].animate.set_color(gold))
        
        # Asset Integration (Issue 19)
        # Using [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/slime.svg]
        slime_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/slime.svg").scale(0.3).set_color(gold)
        # Using [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/mold.svg]
        mold_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/mold.svg").scale(0.3).set_color(gold)
        
        v1_label = MarkupText('V<sub>1</sub>(t)', color=gold).scale(0.8)
        v2_label = MarkupText('V<sub>2</sub>(t)', color=gold).scale(0.8)
        
        v1_group = VGroup(slime_icon, v1_label).arrange(RIGHT, buff=0.1)
        v2_group = VGroup(mold_icon, v2_label).arrange(RIGHT, buff=0.1)
        
        elements = VGroup(v1_group, v2_group).arrange(DOWN, buff=0.4)
        
        # Brackets
        l_bracket = Text("[", font="Consolas", color=gold).scale(2.5)
        r_bracket = Text("]", font="Consolas", color=gold).scale(2.5)
        population_vec = VGroup(l_bracket, elements, r_bracket).arrange(RIGHT, buff=0.1)
        
        v_label_main = MarkupText('<b>V</b>(t) = ', color=gold)
        v_system = VGroup(v_label_main, population_vec).arrange(RIGHT, buff=0.2)
        
        # Fix Issue 23: Positioning
        self.place_in_area(v_system, 'D3', 'E6', scale_factor=1.2)
        
        self.play(FadeIn(v_system))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(white))
        
        matrix_exp_question = MarkupText('e<sup>A</sup> = ?', color=white)
        # Fix Issue 24: Positioning
        self.place_in_area(matrix_exp_question, 'F3', 'F5', scale_factor=1.5)
        
        self.play(Write(matrix_exp_question))
        self.wait(3)
