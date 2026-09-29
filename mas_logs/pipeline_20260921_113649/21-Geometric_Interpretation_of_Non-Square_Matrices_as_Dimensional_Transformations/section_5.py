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
        lecture_lines = ["Non-square matrices connect different dimensions.", "They act as either embeddings or projections.", "These tools are vital for data science."]
        self.setup_layout("Summary & Synthesis", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display a grid summarizing transformation types using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg].
        # (Color: #FFFFFF)
        computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg", color=WHITE)
        self.place_at_grid(computer, "C3", scale_factor=0.5)
        self.play(FadeIn(computer))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        # Flash key concepts like 'Expansion' and 'Compression'. (Color: #FF8080)
        # Fixing issue 33 and issue 38: specific grid anchors
        expansion_label = Text("Expansion", font_size=20, color="#FF8080")
        compression_label = Text("Compression", font_size=20, color="#FF8080")
        self.place_at_grid(expansion_label, "A2", scale_factor=0.6)
        self.place_at_grid(compression_label, "F2", scale_factor=0.6)
        
        self.play(FadeIn(expansion_label), FadeIn(compression_label))
        self.play(self.lecture[1].animate.set_color("#FF8080"))

        # === Animation for Lecture Line 3 ===
        # Show the final mapping diagram overlay using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/projector.svg]. (Color: #80FF80)
        projector = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/projector.svg", color="#80FF80")
        self.place_at_grid(projector, "D4", scale_factor=0.5)
        self.play(FadeIn(projector))
        self.play(self.lecture[2].animate.set_color("#80FF80"))
        
        self.wait(2)
