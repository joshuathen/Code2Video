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
        self.setup_layout("Prerequisite Concept: Embeddings and Vector Space", [
            "Words are converted into multi-dimensional vectors.", 
            "Similar meanings cluster together in space.", 
            "Distances represent semantic relationships between concepts."
        ])
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        map_bg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        
        # === Animation for Lecture Line 1 ===
        # Show a vector as an arrow in #FF4500 from origin, overlaying a compass.svg.
        vec = Vector([1.5, 1.0, 0], color="#FF4500")
        compass_obj = self.place_at_grid(compass, 'C3', scale_factor=0.3)
        self.place_at_grid(vec, 'C3')
        self.play(Create(vec), FadeIn(compass_obj))
        self.lecture[0].set_color("#FF4500")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display multiple vectors clustered in a 2D space colored #32CD32.
        dot_cluster = VGroup()
        for pos in ['B4', 'B5', 'C4', 'C5']:
            dot = Dot(self.grid[pos], color="#32CD32")
            dot_cluster.add(dot)
        self.place_in_area(dot_cluster, 'A4', 'B6', scale_factor=0.7)
        self.play(Create(dot_cluster))
        self.lecture[1].set_color("#32CD32")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight a specific vector point with a blue #1E90FF circle on a background map.svg.
        map_bg_obj = self.place_in_area(map_bg, 'D3', 'F6', scale_factor=0.8)
        selection_circle = Circle(radius=0.2, color="#1E90FF")
        self.place_at_grid(selection_circle, 'A5', scale_factor=0.8)
        
        self.play(FadeIn(map_bg_obj), Create(selection_circle))
        self.lecture[2].set_color("#1E90FF")
        self.wait(2)
