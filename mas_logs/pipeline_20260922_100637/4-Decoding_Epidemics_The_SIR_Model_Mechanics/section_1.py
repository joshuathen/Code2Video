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
        lecture_lines = [
            "Epidemics are modeled by three population states.",
            "Susceptible (S) are healthy individuals.",
            "Infectious (I) are currently carrying the virus.",
            "Recovered (R) have gained immunity.",
            "Total population N equals S plus I plus R."
        ]
        self.setup_layout("The Core Concept: Defining Populations", lecture_lines)
        
        # Mobjects using assets
        s_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg").set_color("#FFD700")
        s_text = Text("S", font_size=24, color=WHITE)
        s_group = VGroup(s_icon, s_text).arrange(DOWN, buff=0.1)
        
        i_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/virus.svg").set_color("#FF4500")
        i_text = Text("I", font_size=24, color=WHITE)
        i_group = VGroup(i_icon, i_text).arrange(DOWN, buff=0.1)
        
        r_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hospital.svg").set_color("#00CED1")
        r_text = Text("R", font_size=24, color=WHITE)
        r_group = VGroup(r_icon, r_text).arrange(DOWN, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(GRAY)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        self.place_in_area(s_group, 'B2', 'B3', scale_factor=0.7)
        self.play(FadeIn(s_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF4500")
        self.place_in_area(i_group, 'B5', 'B6', scale_factor=0.7)
        self.play(FadeIn(i_group))
        
        arrow_si = Arrow(s_group.get_right(), i_group.get_left(), buff=0.1, color=WHITE)
        self.play(Create(arrow_si))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00CED1")
        self.place_in_area(r_group, 'D3', 'D4', scale_factor=0.7)
        self.play(FadeIn(r_group))
        
        arrow_ir = Arrow(i_group.get_bottom(), r_group.get_top(), buff=0.1, color=WHITE)
        self.play(Create(arrow_ir))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        n_text = MathTex("N = S + I + R", font_size=36)
        self.place_in_area(n_text, 'E4', 'F6', scale_factor=0.9)
        self.play(Write(n_text))
        self.wait(2)
