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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines from storyboard
        title_text = "The Great Betrayal (n=6)"
        lecture_lines = [
            "Let's test 6 points and expect 32 regions.",
            "We carefully count every single slice.",
            "Surprisingly, there are only 31 regions!"
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors defined in storyboard
        MAGENTA_COLOR = "#FF00FF"
        WHITE_COLOR = "#FFFFFF"
        RED_COLOR = "#FF0000"
        HIGHLIGHT_COLOR = "#FFFF00"
        CHORD_COLOR = "#58C4DD" 
        
        # === Animation for Lecture Line 1 ===
        # Draw circle with 6 magenta points (#FF00FF) and all chords.
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Fix Issue 28: Reposition circle to B3-E6 and scale by 0.8
        circle = Circle(radius=2.1, color=WHITE_COLOR)
        self.place_in_area(circle, "B3", "E6", scale_factor=0.8)
        center = circle.get_center()
        radius = 2.1 * 0.8
        
        # Define 6 points with non-uniform angles to avoid triple intersections at the center
        # This ensures exactly 31 regions are visible
        angles = [15, 80, 140, 205, 260, 325]
        point_coords = [center + radius * np.array([np.cos(a*DEGREES), np.sin(a*DEGREES), 0]) for a in angles]
        
        dots = VGroup(*[Dot(p, color=MAGENTA_COLOR, radius=0.08) for p in point_coords])
        
        chords = VGroup()
        for i in range(6):
            for j in range(i + 1, 6):
                chords.add(Line(point_coords[i], point_coords[j], color=CHORD_COLOR, stroke_width=1.5))
        
        self.play(Create(circle))
        self.play(FadeIn(dots, shift=UP))
        self.play(Create(chords), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Sequentially number regions from 1 to 31 in white (#FFFFFF).
        self.lecture[0].set_color(WHITE_COLOR)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Map region numbers to the grid layout A1-F6 (36 total cells)
        grid_keys = []
        for r in ["A", "B", "C", "D", "E", "F"]:
            for c in ["1", "2", "3", "4", "5", "6"]:
                grid_keys.append(f"{r}{c}")
        
        # We only need 31 cells for the 31 regions
        number_labels = VGroup()
        for i in range(31):
            num_text = Text(str(i+1), font_size=20, color=WHITE_COLOR)
            self.place_at_grid(num_text, grid_keys[i])
            number_labels.add(num_text)
            
        # Sequentially animate the appearance of each number
        self.play(
            AnimationGroup(
                *[Write(num) for num in number_labels],
                lag_ratio=0.1,
                run_time=3.5
            )
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Surprisingly, there are only 31 regions!
        # Pulse the total count '31' in red (#FF0000) to highlight discrepancy.
        self.lecture[1].set_color(WHITE_COLOR)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Clear the clutter of small numbers and show the final count
        final_count_label = Text("31 Regions", font_size=44, color=RED_COLOR)
        # Fix Issue 29: Position summary label at A5-B6 with scale 0.6
        self.place_in_area(final_count_label, "A5", "B6", scale_factor=0.6)
        
        # Use a background rectangle for better contrast against background/elements
        count_bg = BackgroundRectangle(final_count_label, color=BLACK, fill_opacity=0.85, buff=0.2)
        
        self.play(
            FadeOut(number_labels),
            FadeIn(count_bg),
            Write(final_count_label)
        )
        
        # Pulse animation: scaling up and down
        for _ in range(3):
            self.play(final_count_label.animate.scale(1.2), run_time=0.3)
            self.play(final_count_label.animate.scale(1/1.2), run_time=0.3)
        
        self.wait(2)
