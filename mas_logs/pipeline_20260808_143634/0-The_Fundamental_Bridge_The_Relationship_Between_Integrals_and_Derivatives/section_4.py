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
        self.setup_layout("Synthesis and Real-World Application", [
            "The Fundamental Theorem links these operations.",
            "Integration reconstructs functions from derivatives.",
            "This tool solves real-world physics problems."
        ])
        
        # --- Visual Setup ---
        axes = Axes(x_range=[0, 5, 1], y_range=[0, 4, 1], axis_config={"include_tip": True})
        graph = axes.plot(lambda x: 0.5 * x + 1, x_range=[0, 4], color=BLUE)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/vehicle.svg
        vehicle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vehicle.svg")
        
        total_distance_label = Text("Total Distance", font_size=24, color="#33FF57")
        
        area = axes.get_area(graph, x_range=[0, 4], color=YELLOW, opacity=0.3)
        
        graph_group = VGroup(axes, graph, vehicle)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_in_area(graph_group, 'B2', 'E5', scale_factor=0.9)
        self.place_at_grid(vehicle, 'B2', scale_factor=0.3) # Positioning vehicle near graph
        self.add(axes, graph, vehicle)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.place_in_area(area, 'C3', 'D5', scale_factor=0.7)
        self.play(Create(area))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#33FF57"))
        self.place_at_grid(total_distance_label, 'D4', scale_factor=0.8)
        self.play(Write(total_distance_label))
        self.wait(2)
