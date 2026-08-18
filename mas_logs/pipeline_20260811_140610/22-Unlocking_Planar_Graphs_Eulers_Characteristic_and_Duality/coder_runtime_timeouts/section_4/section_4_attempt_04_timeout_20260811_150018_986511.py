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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Dual graphs mirror original properties.",
            "Original V becomes dual F.",
            "Original E remains dual E.",
            "This creates a perfect inverse symmetry.",
            "A house graph transforms into its dual."
        ]
        self.setup_layout("Duality Properties", lecture_lines)
        
        # Define Colors
        C1 = "#FF9999" # Light Red
        C2 = "#99FF99" # Light Green
        C3 = "#9999FF" # Light Blue
        C4 = "#FFFF99" # Light Yellow
        
        # === Animation for Lecture Line 1 ===
        # Display duality table
        table = Table(
            [["Original", "Dual"], ["V", "F"], ["E", "E"], ["F", "V"]],
            include_outer_lines=True
        ).scale(0.5)
        self.place_in_area(table, "A4", "C6")
        self.play(Create(table), self.lecture[0].animate.set_color(C1))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(C2))
        self.play(Indicate(table.get_entries((2, 1)), color=C2), Indicate(table.get_entries((2, 2)), color=C2))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(C3))
        self.play(Indicate(table.get_entries((3, 1)), color=C3), Indicate(table.get_entries((3, 2)), color=C3))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(C4))
        
        # === Animation for Lecture Line 5 ===
        # Use set_fill and set_color for Polyhedra instead of __init__ arguments
        cube = Cube(side_length=1.5).set_color(WHITE).set_fill(opacity=0.3).rotate(PI/4, axis=OUT)
        octahedron = Octahedron(edge_length=1.2).set_color(C2).set_fill(opacity=0.3)
        
        self.place_in_area(cube, "D4", "F6")
        self.play(Create(cube), self.lecture[4].animate.set_color(WHITE))
        self.play(Transform(cube, octahedron))
        self.wait(2)