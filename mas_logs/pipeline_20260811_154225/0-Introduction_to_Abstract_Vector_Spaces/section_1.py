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
            "Vectors aren't just arrows.",
            "They represent any structured data.",
            "Vectors follow algebraic rules.",
            "Sounds follow those rules too.",
            "Data sets behave like vectors."
        ]
        self.setup_layout("From Concrete to Abstract", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Use simple shapes for vector conceptualization
        vector_arrow = Arrow(start=self.grid["B4"], end=self.grid["D6"], color=WHITE)
        self.play(FadeIn(vector_arrow))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Morph into database icon
        database_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/database.svg", color="#FFFF00")
        self.place_in_area(database_icon, "B4", "E6", scale_factor=0.5)
        self.play(Transform(vector_arrow, database_icon))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Vector Addition rules
        rule_text = MathTex("u + v", color="#00FFFF").scale(0.8)
        self.place_at_grid(rule_text, "C3")
        self.play(Write(rule_text))
        self.lecture[2].set_color("#00FFFF")

        # === Animation for Lecture Line 4 ===
        # Musical Chord
        note_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/note.svg", color="#FF00FF")
        self.place_at_grid(note_icon, "C5", scale_factor=0.7)
        chord_label = Text("Chord", color="#FF00FF").scale(0.7)
        self.place_at_grid(chord_label, "D5")
        self.play(FadeIn(note_icon), Write(chord_label))
        self.lecture[3].set_color("#FF00FF")

        # === Animation for Lecture Line 5 ===
        # Data Set
        dataset_label = Text("Data Set", color="#00FF00").scale(0.7)
        self.place_at_grid(dataset_label, "E5")
        self.play(Write(dataset_label))
        self.lecture[4].set_color("#00FF00")

        self.wait(2)
