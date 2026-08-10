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
        self.setup_layout("Conclusion and Intuition Check", [
            "Counting bounces reveals Pi's hidden digits.", 
            "Discrete collisions meet continuous geometric curves.", 
            "Physics encodes Pi in simple motion."
        ])
        
        # Load assets
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg")
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        
        # Data table for display
        data = [
            ["Mass", "Bounces"],
            ["1", "3"],
            ["100", "31"],
            ["10000", "314"],
            ["1000000", "3141"]
        ]
        
        table = Table(data, include_outer_lines=True)
        self.place_in_area(table, 'C3', 'F5', scale_factor=0.55)
        
        # === Animation for Lecture Line 1 ===
        # Review the key collision steps using the wall asset.
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(wall, 'B2', scale_factor=0.6)
        self.play(FadeIn(wall), FadeIn(table))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Summarize the emergence of Pi.
        self.play(self.lecture[1].animate.set_color("#FF9900"))
        # Highlight specific numbers
        self.add(
            table.add_highlighted_cell((2, 2), color=YELLOW),
            table.add_highlighted_cell((3, 2), color=YELLOW),
            table.add_highlighted_cell((4, 2), color=YELLOW),
            table.add_highlighted_cell((5, 2), color=YELLOW)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final display of the core concept with the sphere asset.
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.place_at_grid(sphere, 'B6', scale_factor=0.6)
        self.play(FadeIn(sphere))
        self.wait(2)
