from manim import *
import numpy as np

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
        # Setup layout with title and lecture lines
        lecture_lines = [
            "Space-filling curves are more than just mathematical curiosities.",
            "They map complex 2D data into a 1D sequence.",
            "Crucially, points near each other in 2D stay nearby.",
            "This locality helps computers process images and GPS data.",
            "Mathematical elegance transforms how we organize our digital world."
        ]
        self.setup_layout("Real-World Application: Organizing Complexity", lecture_lines)

        # Create a 4x4 grid of colored squares on the right side
        grid_squares = VGroup()
        rows_4x4 = ["B", "C", "D", "E"]
        cols_4x4 = ["2", "3", "4", "5"]
        
        # Colors for the grid squares to fulfill "colored squares" requirement
        square_colors = [BLUE_E, GREEN_E, RED_E, PURPLE_E]
        
        for i, r in enumerate(rows_4x4):
            for j, col in enumerate(cols_4x4):
                sq = Square(side_length=0.9, stroke_width=2, color=WHITE, 
                            fill_color=square_colors[(i+j)%4], fill_opacity=0.3)
                self.place_at_grid(sq, f"{r}{col}")
                grid_squares.add(sq)

        # === Animation for Lecture Line 1 ===
        # Show the 4x4 grid of colored squares
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(Create(grid_squares), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Red line row-by-row scan path (#FF0000)
        scan_sequence = []
        for r in rows_4x4:
            for c in cols_4x4:
                scan_sequence.append(f"{r}{c}")
        
        scan_points = [self.grid[pos] for pos in scan_sequence]
        scan_path = VMobject(color="#FF0000")
        scan_path.set_points_as_corners(scan_points)
        
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        self.play(Create(scan_path), run_time=2.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Green Hilbert curve path (#00FF00) for the 4x4 grid
        # Order: (0,0), (1,0), (1,1), (0,1), (0,2), (0,3), (1,3), (1,2), (2,2), (2,3), (3,3), (3,2), (3,1), (2,1), (2,0), (3,0)
        # Corresponding grid keys: B2, C2, C3, B3, B4, B5, C5, C4, D4, D5, E5, E4, E3, D3, D2, E2
        hilbert_sequence = [
            "B2", "C2", "C3", "B3", "B4", "B5", "C5", "C4",
            "D4", "D5", "E5", "E4", "E3", "D3", "D2", "E2"
        ]
        hilbert_points = [self.grid[pos] for pos in hilbert_sequence]
        hilbert_path = VMobject(color="#00FF00")
        hilbert_path.set_points_as_corners(hilbert_points)
        
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        # Transition from the row-scan path to the Hilbert path
        self.play(FadeOut(scan_path), Create(hilbert_path), run_time=2.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight two neighboring squares: B5 and C5 to show Locality
        # They are adjacent in the grid and adjacent in the Hilbert path.
        highlight_b5 = Square(side_length=0.95, color=YELLOW, stroke_width=4)
        highlight_c5 = Square(side_length=0.95, color=YELLOW, stroke_width=4)
        self.place_at_grid(highlight_b5, "B5")
        self.place_at_grid(highlight_c5, "C5")
        
        locality_label = Text("Locality", font_size=24, color="#00FF00")
        self.place_at_grid(locality_label, "A5")
        
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        self.play(Create(highlight_b5), Create(highlight_c5))
        self.play(Write(locality_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Show comparison between Hilbert locality and standard scan distance
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        
        # Temporarily show the old scan path faded out to visualize distance
        faded_scan = scan_path.copy().set_stroke(opacity=0.3)
        scan_dist_arrow = DoubleArrow(self.grid["B5"], self.grid["C5"], color="#FF0000", buff=0.1)
        scan_dist_text = Text("Far in Row-Scan", font_size=18, color="#FF0000")
        self.place_at_grid(scan_dist_text, "C6", scale_factor=0.8) 
        
        self.play(FadeIn(faded_scan))
        self.play(Create(scan_dist_arrow), Write(scan_dist_text))
        self.wait(3)
        
        # Final cleanup
        self.play(FadeOut(faded_scan), FadeOut(scan_dist_arrow), FadeOut(scan_dist_text))
        self.wait(2)
