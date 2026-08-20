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
            "Transformers use Attention and MLP blocks.",
            "Attention links related words together.",
            "MLP layers function as knowledge databases.",
            "Attention acts like a search engine.",
            "MLP stores the factual information."
        ]
        self.setup_layout("Prerequisite: The Transformer Architecture Anatomy", lecture_lines)
        
        # Load Assets
        engine_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/engine.svg")
        database_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/database.svg")
        
        # Create Transformer model representation
        transformer_box = Rectangle(width=3, height=4, color=WHITE)
        attention_box = Rectangle(width=2.5, height=1.5, color=WHITE)
        mlp_box = Rectangle(width=2.5, height=1.5, color=WHITE)
        
        # Apply layout fixes from VideoCritic
        self.place_in_area(transformer_box, 'A2', 'F4', scale_factor=0.85)
        self.place_at_grid(attention_box, 'A4', scale_factor=0.6)
        self.place_at_grid(mlp_box, 'D4', scale_factor=0.6)
        
        # Integrate assets
        self.place_at_grid(engine_icon, 'B4', scale_factor=0.5)
        self.place_at_grid(database_icon, 'E4', scale_factor=0.5)
        
        transformer_group = VGroup(transformer_box, attention_box, mlp_box, engine_icon, database_icon)
        self.play(Create(transformer_group))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(attention_box.animate.set_color("#FFD700"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(mlp_box.animate.set_color("#87CEEB"))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        search_label = Text("Search", font_size=20, color="#00FFFF").next_to(attention_box, RIGHT)
        self.play(Write(search_label))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        database_label = Text("Database", font_size=20, color="#00FFFF").next_to(mlp_box, RIGHT)
        self.play(Write(database_label))
        
        self.wait(2)
