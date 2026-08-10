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
        lecture_lines = ["Transformers have two distinct memory types.", "Attention is your working memory.", "The MLP acts as long-term memory.", "MLPs store facts in synaptic weights.", "It is like a internal brain database."]
        self.setup_layout("The Library vs. The Filing Cabinet", lecture_lines)
        
        # Mobjects
        # Use SVGMobjects for assets
        spotlight_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/spotlight.svg")
        cabinet_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cabinet.svg")
        database_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/database.svg")

        attention_label = Text("Attention", font_size=24)
        attention_group = VGroup(spotlight_asset, attention_label).arrange(DOWN)
        
        mlp_label = Text("MLP", font_size=24)
        mlp_group = VGroup(cabinet_asset, mlp_label).arrange(DOWN)
        
        grid_pattern = VGroup(*[Dot(radius=0.05, color=GREEN_B) for _ in range(20)]).arrange_in_grid(4, 5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_in_area(attention_group, 'B1', 'C2', scale_factor=0.9)
        self.place_in_area(mlp_group, 'B4', 'C5', scale_factor=0.9)
        self.play(Create(attention_group), Create(mlp_group))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        self.play(Indicate(spotlight_asset))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        self.place_in_area(grid_pattern, 'D1', 'F6', scale_factor=0.8)
        self.play(FadeIn(grid_pattern))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(GREEN_B)
        self.play(attention_group.animate.set_opacity(0.5), mlp_group.animate.set_opacity(1.0))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(GOLD)
        self.place_at_grid(database_asset, 'E3', scale_factor=0.5)
        self.play(FadeIn(database_asset), Indicate(attention_group), Indicate(mlp_group))
        self.wait(1)
