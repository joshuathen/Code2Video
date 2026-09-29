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
        lecture_lines = [
            "The windmill combines sorting and incremental updates.",
            "It builds a foundational geometric event sequence.",
            "Essential for advanced algorithms like convex hulls."
        ]
        self.setup_layout("Summary & Complexity Analysis", lecture_lines)
        
        # Elements
        circ = Circle(radius=1.5, color=BLUE).set_stroke(width=3)
        pivot = Dot(color=YELLOW)
        
        # --- Animation for Lecture Line 1 ---
        self.play(FadeIn(circ), FadeIn(pivot))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # --- Animation for Lecture Line 2 ---
        event_line = Line(start=np.array([0,0,0]), end=np.array([1,1,0]), color=RED)
        self.place_at_grid(event_line, 'B4', scale_factor=0.9) # Fix for issue 33
        self.play(Create(event_line))
        self.play(self.lecture[1].animate.set_color(RED))
        
        # --- Animation for Lecture Line 3 ---
        hull = Polygon(np.array([0,1,0]), np.array([1,-0.5,0]), np.array([-1,-0.5,0]), color=GREEN)
        self.place_in_area(hull, 'D3', 'F5', scale_factor=0.7) # Fix for issue 34
        
        # Fix for issue 35: add the missing legend_label
        legend_label = Text("Convex Hull", font_size=20, color=GREEN)
        self.place_at_grid(legend_label, 'F1', scale_factor=0.8)
        
        self.play(DrawBorderThenFill(hull), Write(legend_label))
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.wait(2)
