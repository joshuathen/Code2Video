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

class Section6Scene(TeachingScene):
    def construct(self):
        # Fetching data for the summary section
        lecture_lines = [
            "Change of basis is simply a shift in perspective.",
            "The vector stays fixed while the grid transforms.",
            "Think of it like converting between different currencies."
        ]
        self.setup_layout("Summary & Visual Recap", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        
        # Standard basis grid
        grid_std = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_color": BLUE_E, "stroke_opacity": 0.4},
            axis_config={"stroke_color": BLUE_E}
        )
        # Using a consistent area for the grid display
        self.place_in_area(grid_std, 'B2', 'E5', scale_factor=0.8)
        
        # Fixed vector represented by a currency icon
        # [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/currency.svg]
        currency_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/currency.svg")
        # Place at a fixed position within the visual area
        self.place_at_grid(currency_icon, 'C3', scale_factor=0.5)
        
        self.play(Create(grid_std), FadeIn(currency_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Skewed basis grid (Basis vectors: [1, 0.5] and [0.5, 1])
        matrix = [[1, 0.5], [0.5, 1]]
        grid_skewed = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_color": GOLD_E, "stroke_opacity": 0.6},
            axis_config={"stroke_color": GOLD_E}
        ).apply_matrix(matrix)
        self.place_in_area(grid_skewed, 'B2', 'E5', scale_factor=0.8)
        
        # Matrix labels P and P-1
        p_label = Text("P", color=GREEN, slant=ITALIC)
        p_inv_label = Text("P⁻¹", color=PURPLE, slant=ITALIC)
        
        # [Issue 36 Fix]
        self.place_at_grid(p_label, 'A2', scale_factor=1.0)
        
        # [Issue 37 Fix]
        self.place_at_grid(p_inv_label, 'F6', scale_factor=1.0)
        
        # Arrows bridging the systems
        p_arrow = CurvedArrow(
            self.grid['A3'], 
            self.grid['B5'], 
            angle=-PI/4, 
            color=GREEN
        )
        p_inv_arrow = CurvedArrow(
            self.grid['F5'], 
            self.grid['E2'], 
            angle=-PI/4, 
            color=PURPLE
        )
        
        # Transform the grid background while icon stays fixed
        self.play(
            Transform(grid_std, grid_skewed),
            FadeIn(p_label),
            FadeIn(p_inv_label),
            Create(p_arrow),
            Create(p_inv_arrow),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GOLD)
        
        # Final summary text
        summary_text = Text("Change of Perspective", font_size=24, color="#FFFFFF")
        
        # [Issue 38 Fix]
        self.place_in_area(summary_text, 'F2', 'F5', scale_factor=0.8)
        
        self.play(Write(summary_text))
        self.wait(3)
