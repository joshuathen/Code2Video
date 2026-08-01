from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]
        cols = ["1", "2", "3", "4", "5", "6"]

        right_edge_buffer = 0.5
        grid_cell_spacing = 0.8

        total_grid_width = (len(cols) - 1) * grid_cell_spacing
        total_grid_height = (len(rows) - 1) * grid_cell_spacing

        start_x = (config.frame_width / 2) - (total_grid_width / 2) - right_edge_buffer
        start_y = (total_grid_height / 2)

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = start_x + j * grid_cell_spacing
                y = start_y - i * grid_cell_spacing
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        if grid_pos not in self.grid:
            raise ValueError(f"Grid position '{grid_pos}' not found.")
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left_grid_pos, bottom_right_grid_pos, scale_factor=1.0):
        if top_left_grid_pos not in self.grid or bottom_right_grid_pos not in self.grid:
            raise ValueError(f"Grid position '{top_left_grid_pos}' or '{bottom_right_grid_pos}' not found.")

        tl_pos = self.grid[top_left_grid_pos]
        br_pos = self.grid[bottom_right_grid_pos]

        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])

        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section6Scene(TeachingScene):
    def construct(self):
        title = "Section 6: Privacy-Preserving Contact Tracing Algorithm"
        lecture_points = [
            "- Overview of the algorithm",
            "- Key components and data flow",
            "- Mathematical foundations (if applicable)",
            "- Demonstrating a simplified scenario",
            "- Discussion of privacy guarantees"
        ]
        self.setup_layout(title, lecture_points)

        # Example of placing an object using the grid
        example_text = Text("Algorithm Step 1", font_size=20, color=YELLOW)
        self.play(Write(example_text))
        self.play(example_text.animate.move_to(self.grid["A1"]))
        self.wait(1)

        # Example of placing an object in an area
        area_rect = Rectangle(width=2, height=1, color=BLUE)
        self.play(Create(area_rect))
        self.play(area_rect.animate.move_to(self.place_in_area(area_rect, "C2", "E4").get_center()))
        self.wait(1)

        self.play(FadeOut(example_text), FadeOut(area_rect))