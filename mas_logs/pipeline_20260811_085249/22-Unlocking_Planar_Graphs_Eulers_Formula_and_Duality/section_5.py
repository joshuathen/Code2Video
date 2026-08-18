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
        self.setup_layout("Application & Wrap-up", [
            "Duality helps solve complex routing problems.",
            "Shifting from nodes to regions optimizes paths.",
            "Use dual graphs for efficient network design."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/warehouse.svg]
        # Placing in grid area as requested by critics
        warehouse = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/warehouse.svg")
        warehouse.set_color("#00FFFF")
        self.place_in_area(warehouse, 'C2', 'F5', scale_factor=0.9)
        self.play(Create(warehouse))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        # Animate a robot path using the dual graph
        # Marker scaled down per critic request
        robot = Dot(color="#FFFF00")
        self.place_at_grid(robot, 'C2', scale_factor=0.5)
        path = VMobject(color="#FFFF00").set_points_as_corners([
            self.grid["C2"], self.grid["C4"], self.grid["E4"], self.grid["E2"]
        ])
        self.play(FadeIn(robot), Create(path))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Overlay final application summary text
        summary = Text("Dual Graph Network", font_size=24, color="#FFFFFF")
        self.place_at_grid(summary, 'B2', scale_factor=0.7)
        self.play(FadeIn(summary))
        self.lecture[2].set_color("#FFFFFF")
        self.wait(2)
