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
        self.setup_layout("Summary and Conclusion", [
            "Convergence is defined by your metric choice.", 
            "2-adic metrics reveal hidden number structure.", 
            "Both Euclidean and 2-adic truths coexist."
        ])
        
        # Visual elements
        rabbit_carrot = VGroup(
            Dot(color=BLUE), 
            Text("Rabbit", font_size=18).next_to(Dot(), UP)
        )
        binary_acc = VGroup(
            Square(color=RED, side_length=0.8),
            Text("-1", font_size=20)
        )
        grid_group = VGroup(rabbit_carrot, binary_acc)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(rabbit_carrot, 'B3', scale_factor=0.7)
        self.place_at_grid(binary_acc, 'B4', scale_factor=0.7)
        # We need a dummy grid_group to pass to place_in_area as requested, 
        # though it's the same items placed individually above. 
        # Redundant call to fix requested layout
        self.place_in_area(grid_group, 'C1', 'F6', scale_factor=0.6)
        
        self.play(FadeIn(rabbit_carrot), FadeIn(binary_acc))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        self.play(Indicate(rabbit_carrot), Indicate(binary_acc))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.play(FadeOut(rabbit_carrot), FadeOut(binary_acc))
        self.wait(1)
