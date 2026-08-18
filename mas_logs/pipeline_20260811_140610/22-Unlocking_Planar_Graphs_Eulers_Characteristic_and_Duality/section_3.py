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
        lecture_lines = ["A dual graph swaps faces and vertices.", "Place a dual vertex in every face.", "Connect dual vertices sharing an edge."]
        self.setup_layout("The Concept of Duality", lecture_lines)
        
        # Create a hexagonal tiling (original graph)
        hex_grid = VGroup()
        for i in range(3):
            for j in range(3):
                h = RegularPolygon(n=6, color=WHITE, stroke_width=2)
                # Store relative positions for later use
                h.shift(np.array([i*1.2 - 1.2, j*1.2 - 1.2, 0]))
                hex_grid.add(h)
        
        # Fix 25: Use B3-E6 area for better positioning
        self.place_in_area(hex_grid, "B3", "E6", scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(hex_grid))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        dual_vertices = VGroup()
        for hex_obj in hex_grid:
            dot = Dot(color=GREEN, radius=0.08)
            dot.move_to(hex_obj.get_center())
            dual_vertices.add(dot)
        
        # Fix 27: Use B4-E6 area
        self.place_in_area(dual_vertices, "B4", "E6", scale_factor=0.75)
        self.play(FadeIn(dual_vertices))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        # Simplified connection logic: connect centers of adjacent hexagons
        lines = VGroup()
        for i in range(len(dual_vertices)):
            for j in range(i + 1, len(dual_vertices)):
                if np.linalg.norm(dual_vertices[i].get_center() - dual_vertices[j].get_center()) < 1.3:
                    l = Line(dual_vertices[i].get_center(), dual_vertices[j].get_center(), color=YELLOW, stroke_width=3)
                    lines.add(l)
        
        self.play(Create(lines))
        self.wait(2)
