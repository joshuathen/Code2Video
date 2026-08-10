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
        lecture_lines = [
            "Weights represent sums of outer products.",
            "Training updates these to encode associations.",
            "This forms a grid of knowledge nodes.",
            "We see specific nodes illuminate concepts.",
            "Outer products build the internal map."
        ]
        self.setup_layout("Deep Dive: Rank-One Model Composition", lecture_lines)
        
        # Define colors corresponding to lines
        line_colors = ["#00FFFF", "#FF00FF", "#FFFF00", "#00FF00", "#FFA500"]

        # === Animation for Lecture Line 1 ===
        # Define rank-one matrix as outer product using grid asset.
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        grid_asset.set_color(line_colors[0])
        self.place_at_grid(grid_asset, "B2", scale_factor=0.8)
        self.lecture[0].set_color(line_colors[0])
        self.play(Create(grid_asset))

        # === Animation for Lecture Line 2 ===
        # Show visualization of vector u and v scaling the model output.
        rect = Rectangle(width=2, height=2, color=line_colors[1])
        self.place_at_grid(rect, "B4", scale_factor=0.6)
        self.lecture[1].set_color(line_colors[1])
        self.play(Create(rect))

        # === Animation for Lecture Line 3 ===
        # Use flashing to emphasize the interaction between u and v components.
        dot = Dot(color=line_colors[2])
        self.place_at_grid(dot, "E2", scale_factor=1.5)
        self.lecture[2].set_color(line_colors[2])
        self.play(Flash(dot, color=line_colors[2]))

        # === Animation for Lecture Line 4 ===
        # Assemble the components into a singular matrix structure.
        matrix = Matrix([[1, 0], [0, 1]], v_buff=0.8, h_buff=0.8).set_color(line_colors[3])
        self.place_at_grid(matrix, "D5", scale_factor=0.5)
        self.lecture[3].set_color(line_colors[3])
        self.play(Create(matrix))

        # === Animation for Lecture Line 5 ===
        # Label the resulting rank-one model with map asset.
        map_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        map_asset.set_color(WHITE)
        self.place_at_grid(map_asset, "E3", scale_factor=0.8)
        
        label = Text("Rank-One Model", font_size=24, color=WHITE)
        self.place_at_grid(label, "E3", scale_factor=0.8)
        label.next_to(map_asset, DOWN)
        
        self.lecture[4].set_color(line_colors[4])
        self.play(Create(map_asset), Write(label))
        
        self.wait(2)
