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
        lecture_lines = ["Iteration increases the object's perceived density.", 
                         "Sierpinski triangles leave behind structural voids.", 
                         "Recursive patterns trap area within limited boundaries.", 
                         "Increasing iterations reveals the hidden fractal geometry.", 
                         "Higher dimensions imply more space-filling paths."]
        self.setup_layout("Visualizing Complexity", lecture_lines)
        
        target_icon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/target.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF4500")
        target_icon1 = SVGMobject(target_icon_path, color="#FF4500")
        triangle = Triangle(color="#FF4500").set_fill("#FF4500", opacity=0.5)
        triangle_group = VGroup(triangle, target_icon1).arrange(DOWN)
        self.place_at_grid(triangle_group, "C4", scale_factor=0.7)
        self.play(DrawBorderThenFill(triangle), FadeIn(target_icon1))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#32CD32")
        t1 = Triangle(color="#32CD32").scale(0.5)
        t2 = Triangle(color="#32CD32").scale(0.5)
        t3 = Triangle(color="#32CD32").scale(0.5)
        sierpinski_2 = VGroup(t1, t2, t3).arrange(UP, buff=-0.1).shift(UP*0.3)
        self.place_in_area(sierpinski_2, 'B3', 'C4', scale_factor=0.8)
        self.play(Transform(triangle_group, sierpinski_2))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#1E90FF")
        box = Rectangle(color="#1E90FF", height=1.5, width=1.5)
        self.place_in_area(box, 'B3', 'C4', scale_factor=0.8)
        self.play(Create(box))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFD700")
        eqn = MathTex(r"D = \frac{\log(3)}{\log(2)} \approx 1.58", color="#FFD700")
        self.place_at_grid(eqn, 'D5', scale_factor=0.9)
        self.play(Write(eqn))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF00FF")
        path = Line(start=self.grid["B2"], end=self.grid["E5"], color="#FF00FF")
        terminal_icon = SVGMobject(target_icon_path, color="#FF00FF")
        self.place_at_grid(terminal_icon, 'E5', scale_factor=0.5)
        self.play(Create(path), FadeIn(terminal_icon))
        self.wait(1)
