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
        self.setup_layout("The Core Problem: Contextual Ambiguity", [
            "Traditional models read sequences one by one.",
            "They often lose context for long sentences.",
            "Attention lets models look at all parts."
        ])
        
        # Assets
        book = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg")
        eye = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eye.svg")
        
        self.place_at_grid(book, "B2", scale_factor=1.5)
        self.place_at_grid(eye, "B5", scale_factor=1.5)
        
        sentence = ["The", "animal", "didn't", "cross", "the", "street"]
        word_mobs = VGroup(*[Text(w, font_size=24) for w in sentence])
        for i, mob in enumerate(word_mobs):
            self.place_at_grid(mob, f"D{i+1}")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        for mob in word_mobs:
            self.play(FadeIn(mob), run_time=0.3)
        self.play(Indicate(book))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        self.play(FadeOut(word_mobs[0], word_mobs[1]), run_time=1.0)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        network = VGroup(*[Line(self.grid["D3"], self.grid[f"D{i+1}"], color=YELLOW) for i in range(6)])
        self.play(Create(network), Indicate(eye))
        self.wait(2)
