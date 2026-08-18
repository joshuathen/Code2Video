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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Counter-Intuitive 'Spiky' Volume", [
            "High-dimensional volumes behave quite counter-intuitively.",
            "Volume concentrates primarily near the surface or crust.",
            "The core becomes negligible as dimensions increase.",
            "Imagine peeling a 100-dimensional orange.",
            "Most of the volume resides in the skin."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Visualize high-dimensional volume as a hypercube, color #FF4500.
        hypercube = VGroup(*[Dot(self.grid[pos], color="#FF4500", radius=0.15) for pos in ["B4", "B5", "C4", "C5"]])
        self.place_in_area(hypercube, 'B3', 'C5', scale_factor=1.0)
        self.play(FadeIn(hypercube))
        self.lecture[0].set_color("#FF4500")

        # === Animation for Lecture Line 2 ===
        # Highlight accumulation at corners, color #00FF00.
        corners = VGroup(*[Dot(self.grid[pos], color="#00FF00", radius=0.2) for pos in ["B3", "B6", "E3", "E6"]])
        self.play(Create(corners))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        # Show shrinking core area, color #FF0000.
        core = Circle(radius=0.3, color="#FF0000", fill_opacity=0.5)
        self.place_at_grid(core, "D4")
        self.play(Create(core))
        self.lecture[2].set_color("#FF0000")

        # === Animation for Lecture Line 4 ===
        # Imagine peeling a 100-dimensional orange. Simulate peeling using orange.svg
        orange = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg", color="#FFA500")
        self.place_at_grid(orange, "B4", scale_factor=0.5)
        self.play(FadeIn(orange))
        
        empty_core = Text("Empty Core", font_size=24, color="#FF0000")
        self.place_at_grid(empty_core, "E4", scale_factor=0.8)
        self.play(FadeIn(empty_core))
        self.lecture[3].set_color("#FFA500")

        # === Animation for Lecture Line 5 ===
        # Emphasize volume concentration at surface, color #FFFF00.
        self.play(FadeOut(core), FadeOut(empty_core))
        surface_label = Text("Concentrated at Surface", font_size=24, color="#FFFF00")
        self.place_at_grid(surface_label, "F4", scale_factor=0.9)
        self.play(Write(surface_label))
        self.lecture[4].set_color("#FFFF00")
