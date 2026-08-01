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

class Section1Scene(TeachingScene):
    def construct(self):
        # Assets
        grid_asset = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/grid.svg"
        square_asset = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/square.svg"

        # === Data and Setup ===
        title = "The Search Dilemma: Classical vs. Quantum"
        lecture_lines = [
            "Finding one item in N takes N/2 classical tries.",
            "For 1,000 items, that's about 500 attempts.",
            "Quantum search needs only about 32 tries.",
            "This speedup is known as Grover's Algorithm.",
            "It scales with the square root of N."
        ]
        self.setup_layout(title, lecture_lines)
        
        # === Mobject Preparation ===
        # Grid of 100 white squares [Asset: grid.svg]
        main_grid = SVGMobject(grid_asset).set_color(WHITE)
        self.place_in_area(main_grid, 'B2', 'E5', scale_factor=2.5)
        
        # Individual square highlights coordination
        grid_width = main_grid.width
        grid_height = main_grid.height
        grid_center = main_grid.get_center()
        square_size = (grid_width / 10) * 0.9 
        
        def get_square_pos(i):
            # 10x10 grid logic
            row = i // 10
            col = i % 10
            start_x = grid_center[0] - grid_width/2 + grid_width/20
            start_y = grid_center[1] + grid_height/2 - grid_height/20
            return np.array([start_x + col * (grid_width/10), start_y - row * (grid_height/10), 0])

        # Labels and Formulas (Positioned using areas for better centering)
        classical_label = Text("Classical Search: O(N)", font_size=24, color=WHITE)
        self.place_in_area(classical_label, 'A2', 'A5', scale_factor=0.8)
        
        quantum_label = Text("Quantum Search: O(√N)", font_size=24, color="#00FF00")
        self.place_in_area(quantum_label, 'A2', 'A5', scale_factor=0.8)
        
        formula = Text("√1000 ≈ 32", font_size=36, color="#00FF00")
        self.place_in_area(formula, 'F2', 'F5', scale_factor=1.0)
        
        # Scan line for quantum processing
        scan_line = Line(
            main_grid.get_top() + UP*0.2, 
            main_grid.get_bottom() + DOWN*0.2, 
            color="#00FFFF", 
            stroke_width=4
        )
        scan_line.set_x(main_grid.get_left()[0])

        # === Animation for Lecture Line 1 ===
        # "Finding one item in N takes N/2 classical tries."
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(
            DrawBorderThenFill(main_grid),
            FadeIn(classical_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "For 1,000 items, that's about 500 attempts."
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        # Sequentially highlight 50 squares [Asset: square.svg] in red
        highlights = VGroup()
        highlight_anims = []
        for i in range(50):
            sq = SVGMobject(square_asset).set_fill("#FF0000", opacity=1).set_color("#FF0000").scale_to_fit_width(square_size)
            sq.move_to(get_square_pos(i))
            highlights.add(sq)
            highlight_anims.append(FadeIn(sq))
            
        self.play(LaggedStart(*highlight_anims, lag_ratio=0.04, run_time=3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Quantum search needs only about 32 tries."
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        # Clear red highlights and switch labels
        self.play(
            FadeOut(classical_label),
            FadeIn(quantum_label),
            FadeOut(highlights),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "This speedup is known as Grover's Algorithm."
        self.play(self.lecture[3].animate.set_color("#00FFFF"))
        # Fast cyan scan line passing across the grid [Asset: grid.svg]
        self.add(scan_line)
        self.play(
            scan_line.animate.set_x(main_grid.get_right()[0]), 
            run_time=1.5, 
            rate_func=linear
        )
        self.play(FadeOut(scan_line))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "It scales with the square root of N."
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        # Highlight a single target square [Asset: square.svg] in bright green
        target_idx = 42
        target_sq = SVGMobject(square_asset).set_fill("#00FF00", opacity=1).set_color("#00FF00").scale_to_fit_width(square_size)
        target_sq.move_to(get_square_pos(target_idx))
        
        self.play(
            FadeIn(target_sq),
            FadeIn(formula),
            run_time=1.5
        )
        self.wait(2)
