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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Gravity converts potential energy into kinetic energy.", 
                         "Velocity increases with the square root of depth.", 
                         "A particle accelerates as it moves downward."]
        self.setup_layout("Prerequisite Intuition: Energy Conservation", lecture_lines)
        
        # Elements
        bar_container = Rectangle(width=0.6, height=3.2, color=WHITE)
        pe_bar = Rectangle(width=0.6, height=3.2, color=BLUE, fill_opacity=0.6)
        ke_bar = Rectangle(width=0.6, height=0.0, color=RED, fill_opacity=0.6)
        
        # Group energy system
        energy_system = VGroup(bar_container, pe_bar, ke_bar)
        
        # Positioning using place_in_area as requested
        self.place_in_area(energy_system, 'B3', 'E5', scale_factor=0.5)
        
        pe_label = Text("PE", font_size=18, color=BLUE)
        ke_label = Text("KE", font_size=18, color=RED)
        self.place_at_grid(pe_label, 'C3', scale_factor=0.5)
        self.place_at_grid(ke_label, 'C5', scale_factor=0.5)
        
        particle = Dot(color=YELLOW, radius=0.15)
        self.place_at_grid(particle, 'B4', scale_factor=0.6)
        
        # Tracker for animation
        depth = ValueTracker(0)
        
        # Updater
        def update_bars(mob):
            d = depth.get_value()
            pe_bar.set_height(3.2 * (1 - d/3.0), stretch=True)
            pe_bar.align_to(bar_container, DOWN)
            ke_bar.set_height(3.2 * (d/3.0), stretch=True)
            ke_bar.align_to(bar_container, UP)
        
        energy_system.add_updater(update_bars)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(bar_container), FadeIn(pe_bar), FadeIn(ke_bar), FadeIn(pe_label), FadeIn(ke_label), Create(particle))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        self.play(depth.animate.set_value(1.5), particle.animate.move_to(self.grid["D4"]), run_time=2)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        self.play(depth.animate.set_value(3.0), particle.animate.move_to(self.grid["E4"]), run_time=2)
        
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
