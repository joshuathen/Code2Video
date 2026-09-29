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
        self.setup_layout("Visualizing the Flow: The SIR Engine", [
            "States behave like connected fluid tanks.",
            "Flow rates represent transition speeds.",
            "Contact rates drive Susceptible to Infected.",
            "Recovery rates move Infected to Recovered.",
            "SIR_Tank_Animation illustrates this movement."
        ])

        # Define containers
        s_box = Square(side_length=1.5, color="#0000FF", fill_opacity=0.3)
        i_box = Square(side_length=1.5, color="#FF0000", fill_opacity=0.3)
        r_box = Square(side_length=1.5, color="#00FF00", fill_opacity=0.3)
        
        s_label = Text("S", color="#0000FF")
        i_label = Text("I", color="#FF0000")
        r_label = Text("R", color="#00FF00")

        # Load SVG Asset
        tank_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tank.svg")

        tanks = VGroup(s_box, i_box, r_box).arrange(RIGHT, buff=0.5)
        SIR_Engine_group = VGroup(tanks, s_label, i_label, r_label, tank_icon)
        
        # Applying positioning constraints
        self.place_in_area(tanks, 'B4', 'D6', scale_factor=0.55)
        self.place_at_grid(s_label, 'B3', scale_factor=0.7)
        self.place_at_grid(i_label, 'C3', scale_factor=0.7)
        self.place_at_grid(r_label, 'D3', scale_factor=0.7)
        self.place_in_area(SIR_Engine_group, 'B4', 'E6', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#ADD8E6")
        self.play(FadeIn(tanks), FadeIn(s_label), FadeIn(i_label), FadeIn(r_label))

        # === Animation for Lecture Line 2 ===
        self.wait(1)
        self.lecture[1].set_color("#FFB6C1")

        # === Animation for Lecture Line 3 ===
        self.wait(1)
        self.lecture[2].set_color("#0000FF")
        arrow = Arrow(s_box.get_right(), i_box.get_left(), color=WHITE)
        self.play(Create(arrow))
        
        # === Animation for Lecture Line 4 ===
        self.wait(1)
        self.lecture[3].set_color("#FF0000")
        arrow2 = Arrow(i_box.get_right(), r_box.get_left(), color=WHITE)
        self.play(Create(arrow2))

        # === Animation for Lecture Line 5 ===
        self.wait(1)
        self.lecture[4].set_color("#90EE90")
        self.play(FadeOut(arrow), FadeOut(arrow2), FadeIn(tank_icon))
        
        dots = VGroup(*[Dot(point=s_box.get_center(), color=WHITE) for _ in range(5)])
        self.play(FadeIn(dots))
        self.play(dots.animate.move_to(i_box.get_center()), run_time=2)
        self.play(dots.animate.move_to(r_box.get_center()), run_time=2)
        self.wait(2)
