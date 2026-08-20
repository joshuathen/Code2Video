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
        lecture_lines = [
            "Bayes' theorem quantifies how we learn from data.",
            "Independence tells us when learning from data fails.",
            "Use these tools to navigate predictive uncertainty."
        ]
        self.setup_layout("Practical Application Summary", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Fade in list of concepts with notebook icon
        icon_notebook = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/notebook.svg", color=WHITE)
        concept_list = VGroup(
            Text("Bayes' Theorem", font_size=24, color=WHITE),
            Text("Independence", font_size=24, color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT)
        
        concept_group = VGroup(icon_notebook, concept_list).arrange(RIGHT, buff=0.3)
        self.place_in_area(concept_group, 'B4', 'C5', scale_factor=0.9)
        
        self.play(FadeIn(concept_group))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight independence and Bayes connection
        self.play(
            concept_list[0].animate.set_color("#FFD700"),
            concept_list[1].animate.set_color("#FFD700"),
            self.lecture[1].animate.set_color("#FFD700")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show summary text box with calculator icon
        icon_calc = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg", color="#32CD32")
        summary_text = Text("Predictive Logic", font_size=24, color="#32CD32")
        summary_group = VGroup(icon_calc, summary_text).arrange(RIGHT, buff=0.3)
        
        box = SurroundingRectangle(concept_list, color="#32CD32", buff=0.3)
        
        # Position summary group below the box
        self.place_at_grid(summary_group, 'E4', scale_factor=0.8)
        
        self.play(Create(box), Write(summary_group))
        self.play(self.lecture[2].animate.set_color("#32CD32"))
        self.wait(2)
