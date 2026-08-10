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
        lecture_lines = ["Independence simplifies computation significantly.", 
                         "Bayes' Theorem provides the logic for learning.", 
                         "Together, they power modern data classifiers."]
        self.setup_layout("Summary & Application Loop", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Load asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        self.place_at_grid(computer_icon, "B2", scale_factor=0.6)
        self.play(FadeIn(computer_icon))
        self.lecture[0].set_color("#FF00FF")

        # === Animation for Lecture Line 2 ===
        # Highlight Bayes Theorem (#FFFF00)
        self.lecture[1].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show final synthesis formula on screen with the [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg]
        server_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg")
        self.place_at_grid(server_icon, "D4", scale_factor=0.6)
        formula = MathTex(r"P(S|W) = \frac{P(W|S)P(S)}{P(W)}").set_color(WHITE)
        self.place_at_grid(formula, "E4", scale_factor=0.8)
        
        self.play(FadeIn(server_icon), Write(formula))
        self.lecture[2].set_color("#FFFFFF")
        self.wait(2)
