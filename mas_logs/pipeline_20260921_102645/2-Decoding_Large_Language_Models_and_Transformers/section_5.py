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
        self.setup_layout("Real-world Application: Generative AI", [
            "LLMs generate content by sampling probabilities.",
            "Models evaluate the next most likely token.",
            "Creative generation results from iterative prediction."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        self.place_at_grid(computer, 'B3', scale_factor=1.5)
        self.play(FadeIn(computer))
        
        prob_dist = VGroup(
            Text("salt: 60%", font_size=18),
            Text("sugar: 40%", font_size=18)
        ).arrange(DOWN)
        self.place_at_grid(prob_dist, 'E3', scale_factor=1.0)
        self.play(Write(prob_dist))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        highlight = SurroundingRectangle(prob_dist[0], color="#FFFF00", buff=0.1)
        self.play(Create(highlight))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        server = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg")
        self.place_at_grid(server, 'B5', scale_factor=1.5)
        self.play(FadeIn(server))
        
        sentence = Text("Cooking with salt...", font_size=20)
        self.place_at_grid(sentence, 'E5', scale_factor=1.0)
        self.play(Write(sentence))
        self.wait(2)
