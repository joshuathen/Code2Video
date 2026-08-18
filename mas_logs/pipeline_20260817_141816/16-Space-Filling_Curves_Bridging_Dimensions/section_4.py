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
        self.setup_layout("Real-World Applications", [
            "Map 2D coordinates into a 1D sequence.",
            "Useful for image processing and database indexing.",
            "Optimizes storage for high-speed retrieval tasks."
        ])
        
        # Animations
        # === Animation for Lecture Line 1 ===
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg", color="#FF0000")
        self.place_at_grid(camera_icon, 'B4', scale_factor=0.7)
        pixel_label = Text("Pixel Map", font_size=20, color="#FF0000")
        self.place_at_grid(pixel_label, 'A4', scale_factor=0.7)
        
        self.play(FadeIn(camera_icon), Write(pixel_label))
        self.lecture[0].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        cache_line = RoundedRectangle(corner_radius=0.1, height=0.5, width=2, color="#00FF00")
        self.place_at_grid(cache_line, 'D4', scale_factor=0.7)
        cache_label = Text("Cache Line", font_size=20, color="#00FF00")
        self.place_at_grid(cache_label, 'C4', scale_factor=0.7)
        
        self.play(FadeIn(cache_line), Write(cache_label))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        monitor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/monitor.svg", color="#0000FF")
        self.place_at_grid(monitor_icon, 'F5', scale_factor=0.7)
        query_label = Text("Spatial Query", font_size=20, color="#0000FF")
        self.place_at_grid(query_label, 'E5', scale_factor=0.7)
        
        self.play(FadeIn(monitor_icon), Write(query_label))
        self.lecture[2].set_color("#0000FF")
        self.wait(2)
