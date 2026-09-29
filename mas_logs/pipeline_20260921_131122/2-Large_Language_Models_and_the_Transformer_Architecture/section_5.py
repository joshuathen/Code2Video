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
        self.setup_layout("Summary & Application", [
            "Vectors, Attention, and Scale work together.", 
            "The model calculates next-word probabilities.", 
            "It generates text by traversing space."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display the three concepts (Vectors, Attention, Scale) merging into one.
        txt1 = Text("Vectors", color=BLUE, font_size=24)
        txt2 = Text("Attention", color=GREEN, font_size=24)
        txt3 = Text("Scale", color=RED, font_size=24)
        self.place_at_grid(txt1, "B1")
        self.place_at_grid(txt2, "B3")
        self.place_at_grid(txt3, "B5")
        
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(txt1), FadeIn(txt2), FadeIn(txt3))
        combined = VGroup(txt1, txt2, txt3).animate.move_to(self.grid["B3"])
        self.play(combined)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show a probability bar graph evolving
        graph = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        self.place_at_grid(graph, "D3", scale_factor=0.5)
        
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(FadeIn(graph))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Trace a path through the vector space
        path = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/path.svg")
        self.place_at_grid(path, "E4", scale_factor=0.5)
        
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(FadeIn(path))
        self.wait(2)
