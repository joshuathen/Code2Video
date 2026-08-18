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
        self.setup_layout("Key Takeaways & Real-World Application", 
                          ["Gaussians are closed under addition.", 
                           "Errors accumulate but retain the Gaussian family.", 
                           "Essential for combining sensor data."])
        
        # === Animation for Lecture Line 1 ===
        # Display bullet point list #FFFFFF of takeaways.
        takeaways = VGroup(
            Text("Closure under Addition", font_size=24, color="#FFFFFF"),
            Text("Gaussian Family Persistence", font_size=24, color="#FFFFFF")
        ).arrange(DOWN, buff=0.5)
        self.place_in_area(takeaways, 'A2', 'C4', scale_factor=0.75)
        
        # Add Sensor Fusion Utility separately to handle positioning fix (Issue 32)
        sensor_text = Text("Sensor Fusion Utility", font_size=24, color="#FFFFFF")
        self.place_at_grid(sensor_text, 'B5', scale_factor=0.8)
        
        self.play(Write(takeaways), Write(sensor_text))

        # === Animation for Lecture Line 2 ===
        # Highlight #FFFF00 key application in signal processing using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg].
        sensor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg")
        self.place_at_grid(sensor_icon, 'E5', scale_factor=0.5)
        
        self.play(
            self.lecture[2].animate.set_color("#FFFF00"),
            sensor_text.animate.set_color("#FFFF00"),
            FadeIn(sensor_icon)
        )

        # === Animation for Lecture Line 3 ===
        # Display final conclusion text #FFFFFF.
        conclusion = Text("Real-world Reliability", font_size=32, color="#FFFFFF")
        self.place_at_grid(conclusion, 'D4', scale_factor=0.8)
        self.play(FadeIn(conclusion))
        self.wait(2)
