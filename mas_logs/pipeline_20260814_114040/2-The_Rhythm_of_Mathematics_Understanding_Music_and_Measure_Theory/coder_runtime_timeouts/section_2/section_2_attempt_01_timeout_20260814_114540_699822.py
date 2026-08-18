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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Measures organize beats into defined containers.", 
                         "Time signatures define the capacity of each measure.", 
                         "Think of it as a musical basket."]
        self.setup_layout("Defining the Measure (The Container)", lecture_lines)
        
        # Elements
        container = Square(side_length=2.5, color="#FFA500")
        container_label = Text("Container", font_size=24, color="#FFA500")
        
        # Trackable values
        m_val = ValueTracker(0)
        
        # Circle updater
        circles = VGroup()
        def update_circles(group):
            count = int(m_val.get_value())
            new_circles = VGroup(*[Dot(color=YELLOW) for _ in range(count)])
            new_circles.arrange_in_grid(buff=0.1)
            new_circles.scale_to_fit_width(container.width * 0.8)
            new_circles.move_to(container.get_center())
            group.become(new_circles)
            
        circles.add_updater(update_circles)
        
        # Label m
        m_text = Text("m =", font_size=24, color="#00FF00")
        m_number = DecimalNumber(0, num_decimal_places=0, color="#00FF00")
        m_group = VGroup(m_text, m_number).arrange(RIGHT)
        m_number.add_updater(lambda d: d.set_value(m_val.get_value()))
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFA500")
        self.place_at_grid(container, 'C2')
        container_label.next_to(container, UP)
        self.add(container, container_label)
        self.wait(3)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        self.place_at_grid(m_group, 'C5')
        self.add(m_group, circles)
        self.play(m_val.animate.set_value(4), run_time=2)
        self.wait(3)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        # Animate basket expansion
        self.play(
            container.animate.scale(1.2),
            container_label.animate.scale(1.2),
            run_time=2
        )
        self.wait(3)
