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
            "The population is split into three compartments.",
            "Susceptible, Infectious, and Recovered states.",
            "The model tracks transitions between these states."
        ]
        self.setup_layout("Defining the Components: The S-I-R Framework", lecture_lines)

        # Load Assets
        person_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg")
        hospital_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hospital.svg")

        # Define Nodes
        s_node = Circle(radius=0.5, color="#0000FF", fill_opacity=0.5)
        s_text = Tex("S", color=WHITE).move_to(s_node.get_center())
        s_group = VGroup(s_node, s_text, person_icon.copy().scale(0.3).next_to(s_node, DOWN, buff=0.1))

        i_node = Circle(radius=0.5, color="#FF0000", fill_opacity=0.5)
        i_text = Tex("I", color=WHITE).move_to(i_node.get_center())
        i_group = VGroup(i_node, i_text, person_icon.copy().scale(0.3).next_to(i_node, DOWN, buff=0.1))

        r_node = Circle(radius=0.5, color="#00FF00", fill_opacity=0.5)
        r_text = Tex("R", color=WHITE).move_to(r_node.get_center())
        r_group = VGroup(r_node, r_text, person_icon.copy().scale(0.3).next_to(r_node, DOWN, buff=0.1))

        flow_arrow_si = Arrow(s_group.get_right(), i_group.get_left(), color=WHITE)
        flow_arrow_ir = Arrow(i_group.get_right(), r_group.get_left(), color=WHITE)
        
        # Add hospital icon near S-I transition
        hospital_icon.scale(0.5)
        self.place_at_grid(hospital_icon, 'E4')

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.place_at_grid(s_group, 'C2', scale_factor=0.8)
        self.place_at_grid(i_group, 'C4', scale_factor=0.8)
        self.place_at_grid(r_group, 'C6', scale_factor=0.8)
        self.play(FadeIn(s_group, i_group, r_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFD700")
        self.play(Create(flow_arrow_si), Create(flow_arrow_ir))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFD700")
        self.play(Indicate(flow_arrow_si, color="#FF9900"))
        self.play(FadeIn(hospital_icon))
        self.wait(1)
