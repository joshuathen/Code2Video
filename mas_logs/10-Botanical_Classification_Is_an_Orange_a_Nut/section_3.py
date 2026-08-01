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
        # Setup the title and lecture lines
        self.setup_layout("Anatomy of an Orange: The Hesperidium", [
            "An orange is a fleshy fruit called a hesperidium.",
            "The outer peel is the protective exocarp.",
            "The white pith inside is called the mesocarp.",
            "The endocarp contains the segments we eat.",
            "Inside segments are tiny, liquid-filled juice vesicles."
        ])

        # === Animation for Lecture Line 1 ===
        # An orange is a fleshy fruit called a hesperidium.
        self.play(self.lecture[0].animate.set_color("#FFA500"))
        
        # Create base orange layers
        peel = Annulus(inner_radius=1.1, outer_radius=1.2, color="#FFA500", fill_opacity=1)
        pith = Annulus(inner_radius=1.0, outer_radius=1.1, color="#FFA500", fill_opacity=1)
        
        num_segments = 10
        segments = VGroup()
        for i in range(num_segments):
            angle_start = i * (TAU / num_segments) + 0.05
            angle_sweep = (TAU / num_segments) - 0.1
            seg = AnnularSector(
                inner_radius=0.1, 
                outer_radius=0.95, 
                angle=angle_sweep, 
                start_angle=angle_start, 
                color="#FFA500", 
                fill_opacity=1
            )
            segments.add(seg)
            
        orange_group = VGroup(peel, pith, segments)
        # Place orange at D3 (central-ish area for the right side)
        self.place_at_grid(orange_group, "D3", scale_factor=0.8)
        
        self.play(FadeIn(orange_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The outer peel is the protective exocarp.
        self.play(self.lecture[1].animate.set_color("#FF8C00"))
        exocarp_label = Text("Exocarp", font_size=20, color="#FF8C00")
        # Position label above the orange to avoid obstruction
        self.place_at_grid(exocarp_label, "B3")
        
        self.play(
            peel.animate.set_color("#FF8C00"),
            Write(exocarp_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The white pith inside is called the mesocarp.
        self.play(self.lecture[2].animate.set_color("#F5F5DC"))
        mesocarp_label = Text("Mesocarp", font_size=20, color="#F5F5DC")
        # Position label to the right of the orange
        self.place_at_grid(mesocarp_label, "D5")
        
        self.play(
            pith.animate.set_color("#F5F5DC"),
            Write(mesocarp_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The endocarp contains the segments we eat.
        self.play(self.lecture[3].animate.set_color("#FFD700"))
        endocarp_label = Text("Endocarp", font_size=20, color="#FFD700")
        # Position label below the orange
        self.place_at_grid(endocarp_label, "F3")
        
        self.play(
            segments.animate.set_color("#FFD700"),
            Write(endocarp_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Inside segments are tiny, liquid-filled juice vesicles.
        self.play(self.lecture[4].animate.set_color("#FFFFE0"))
        
        # Magnified bubble for juice vesicles in the top-right
        bubble = Circle(radius=0.6, color=WHITE, fill_opacity=0.1)
        self.place_at_grid(bubble, "B5")
        
        # Create small liquid-filled balloons (vesicles)
        vesicles = VGroup()
        for _ in range(10):
            v = Ellipse(width=0.1, height=0.2, color="#FFFFE0", fill_opacity=0.8)
            # Random position inside bubble center
            offset = np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3), 0])
            v.move_to(bubble.get_center() + offset)
            v.rotate(np.random.uniform(0, TAU))
            vesicles.add(v)
            
        vesicle_label = Text("Juice Vesicles", font_size=20, color="#FFFFE0")
        self.place_at_grid(vesicle_label, "A5")
        
        # Line from one of the segments to the bubble
        zoom_line = Line(
            orange_group.get_center() + UP*0.4 + RIGHT*0.4, 
            bubble.get_center(), 
            stroke_width=2, 
            color=GRAY
        )
        
        self.play(
            Create(zoom_line),
            FadeIn(bubble),
            FadeIn(vesicles),
            Write(vesicle_label)
        )
        self.wait(2)
