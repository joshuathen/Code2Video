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
        self.setup_layout("Synthesis and Summary", [
            "Forward prediction calculates the current error.",
            "Gradient descent updates parameters to reduce it.",
            "Repeating this process perfects the model."
        ])
        
        # Define loop elements
        nodes = ["Predict", "Cost", "Gradient", "Update"]
        node_objs = VGroup(*[VGroup(Circle(radius=0.5, color="#FFFFFF"), Text(n, font_size=16, color=WHITE)) for n in nodes])
        for i in range(len(node_objs)):
            node_objs[i][1].move_to(node_objs[i][0])
            
        self.place_at_grid(node_objs[0], 'C4', scale_factor=0.7)
        self.place_at_grid(node_objs[1], 'C5', scale_factor=0.7)
        self.place_at_grid(node_objs[2], 'D5', scale_factor=0.7)
        self.place_at_grid(node_objs[3], 'D4', scale_factor=0.7)
        
        arrows = VGroup(
            Arrow(node_objs[0].get_right(), node_objs[1].get_left(), color="#FFFFFF", buff=0.1),
            Arrow(node_objs[1].get_bottom(), node_objs[2].get_top(), color="#FFFFFF", buff=0.1),
            Arrow(node_objs[2].get_left(), node_objs[3].get_right(), color="#FFFFFF", buff=0.1),
            Arrow(node_objs[3].get_top(), node_objs[0].get_bottom(), color="#FFFFFF", buff=0.1)
        )
        
        # Terminal icon asset
        terminal = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/terminal.svg", color="#00FF00")
        self.place_at_grid(terminal, 'C3', scale_factor=0.3)
        
        self.add(node_objs, arrows, terminal)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"), run_time=1)
        self.play(Indicate(node_objs[0]), Indicate(node_objs[1]), run_time=1.5)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"), self.lecture[1].animate.set_color("#FFFF00"), run_time=1)
        self.play(Indicate(node_objs[2]), Indicate(node_objs[3]), run_time=1.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"), self.lecture[2].animate.set_color("#FFFF00"), run_time=1)
        # Rotate entire group around center
        center_point = (self.grid['C4'] + self.grid['C5'] + self.grid['D5'] + self.grid['D4']) / 4
        self.play(Rotate(node_objs, angle=2*PI, about_point=center_point), run_time=2)
        self.play(Flash(terminal, color="#00FF00"), run_time=1)
