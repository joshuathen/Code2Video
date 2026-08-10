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
            "MLP layers function as a key-value memory system.",
            "The first layer acts as a pattern matcher.",
            "The second layer projects information to the output space.",
            "Input 'France' activates specific key-value pathways.",
            "Resulting output retrieves the concept 'Paris'."
        ]
        self.setup_layout("The Anatomy of an MLP: Key-Value Pairs", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        rect = Square(side_length=1.5, color="#FF5733")
        label = Text("MLP", font_size=24)
        mlp_block = VGroup(rect, label)
        self.place_at_grid(mlp_block, "C2", scale_factor=0.7)
        france_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/france.svg")
        self.place_at_grid(france_icon, "A2", scale_factor=0.5)
        self.play(FadeIn(mlp_block), FadeIn(france_icon))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#33FF57")
        matcher_box = Rectangle(width=2, height=0.5, color="#33FF57")
        matcher_label = Text("Pattern Matcher", font_size=18)
        matcher = VGroup(matcher_box, matcher_label)
        self.place_at_grid(matcher, "C4", scale_factor=0.7)
        self.play(FadeIn(matcher))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#3357FF")
        projector_box = Rectangle(width=2, height=0.5, color="#3357FF")
        projector_label = Text("Projector", font_size=18)
        projector = VGroup(projector_box, projector_label)
        self.place_at_grid(projector, "C5", scale_factor=0.7)
        self.play(FadeIn(projector))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF33")
        input_text = Text("France", font_size=24, color="#FFFF33")
        self.place_at_grid(input_text, "B1", scale_factor=0.7)
        arrow = Arrow(start=input_text.get_right(), end=matcher.get_left(), color="#FFFF33")
        self.play(FadeIn(input_text), GrowArrow(arrow))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF33A8")
        output_text = Text("Paris", font_size=24, color="#FF33A8")
        self.place_at_grid(output_text, "D6", scale_factor=0.7)
        paris_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/paris.svg")
        self.place_at_grid(paris_icon, "E6", scale_factor=0.5)
        arrow2 = Arrow(start=projector.get_right(), end=output_text.get_left(), color="#FF33A8")
        self.play(FadeIn(output_text), FadeIn(paris_icon), GrowArrow(arrow2))
        
        self.wait(2)
