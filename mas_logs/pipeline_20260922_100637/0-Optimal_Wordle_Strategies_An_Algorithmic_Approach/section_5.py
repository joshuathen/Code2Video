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
        self.setup_layout("Simulation & Conclusion", [
            "Combine high-entropy opens with pruning.",
            "Computer bots minimize path length effectively.",
            "Optimal play beats human intuition."
        ])
        
        # --- Visual Setup ---
        # Simulation space
        sim_box = Rectangle(width=4, height=4, color=BLUE).move_to(self.grid["C4"])
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Simulate word list guesses
        sim_text = Text("Simulating Guesses...", font_size=24)
        self.place_at_grid(sim_text, 'B3', scale_factor=0.8)
        self.place_at_grid(computer_icon, 'C4', scale_factor=0.5)
        
        self.play(Create(sim_box), Write(sim_text), FadeIn(computer_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Display path length
        path_label = Text("Path Length: 3", font_size=24, color=WHITE)
        self.place_at_grid(path_label, 'E3', scale_factor=0.7)
        
        self.play(FadeOut(sim_text), FadeOut(computer_icon), Write(path_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Show final success distribution
        # BarChart in Manim CE
        dist = BarChart(values=[0.1, 0.3, 0.4, 0.15, 0.05], y_range=[0, 0.5, 0.1], bar_names=["1", "2", "3", "4", "5"])
        self.place_in_area(dist, 'B2', 'D5', scale_factor=0.9)
        self.play(FadeOut(sim_box), FadeOut(path_label), Create(dist))
        self.wait(2)
